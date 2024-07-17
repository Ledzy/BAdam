import torch
import random
from torch.optim import Optimizer
from torch import Tensor
from collections import defaultdict
from typing import List, Optional, Dict, Union, Iterable
import time
import math
import warnings
import gc
import re
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from deepspeed.runtime.zero.utils import apply_to_tensors_only
from deepspeed.utils import z3_leaf_parameter
from deepspeed.utils import logger
from transformers.integrations import is_deepspeed_zero3_enabled
import logging

logger.setLevel(logging.WARNING) # surpress the tedious info log from deepspeed when switching trainable blocks

# Optional [0, 1, 2]. 
    # 0: no print
    # 1: print the relative time whenever a parameter's grad is ready
    # 2: for debug usage only. Will set all the parameters trainable, print the grad ready time for each parameter. 
    #     In this case, all the grad except the "specified" trainable parameters will be set to None after being calculated.
BACKWARD_VERBOSE = 0

def print_rank_0(s, force=True):
    if not torch.distributed.is_initialized():
        print(s)
    elif torch.distributed.get_rank() == 0 and force:
        print(s)

class BlockOptimizer(Optimizer):
    """Wrap the original optimizer to update trainable parameters periodically based on a specified block list."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        named_parameters_list,
        block_prefix_list: List[str] = None,
        switch_block_every: int = 50,
        start_block: Optional[int] = None,
        switch_mode: str = "descending",
        active_modules: List[str] = [],
        include_embedding=False,
        include_lm_head=False,
        verbose: int = 1,
        log_fn = None,
        ds_zero3_enabled = None
    ):
        """
        Args:
            base_optimizer (Optimizer): The base optimizer being wrapped by the BlockOptimizer.
            named_parameters_list: A function that generates the named parameters of the model.
            block_prefix_list (List[List[str]]): The list of blocks of parameters to be updated.
            switch_block_every (int, optional): The number of optimization steps before switching to the next block. Defaults to 10.
            start_block (Optional[int], optional): The index of the block to start with. Defaults to None.
            switch_mode (str, optional): The mode for switching between different blocks of parameters. Defaults to "descending".
            active_modules (List[str]): The list of modules that are always active during optimization. Defaults to None.
            verbose (int, optional): The verbosity level for printing information during optimization. Defaults to 1.
            log_fn: A logging function for recording information during optimization. Defaults to None.
            ds_zero3_enabled: Whether to use DeepSpeed ZeRO-3.
        """
        if block_prefix_list is None:
            block_prefix_list = self.infer_param_groups([n for n, _ in named_parameters_list], include_embedding, include_lm_head)

        assert switch_mode in ["random", "descending", "ascending", "fixed"]
        assert isinstance(block_prefix_list, list)
        if ds_zero3_enabled is not None:
            logger.warning("ds_zero3_enabled is deprecated and is not used anymore. BAdam will automatically detects if DeepSpeed ZeRO-3 is enabled.")

        self.verbose = verbose
        self.switch_mode = switch_mode
        self.switch_block_every = switch_block_every
        self.named_parameters_list = named_parameters_list
        self.weight_decay = base_optimizer.param_groups[0]["weight_decay"]
        self.block_prefix_list = block_prefix_list
        self.block_num = len(block_prefix_list)
        self.log_fn = log_fn
        self.global_step = 0
        self.base_optimizer = base_optimizer
        self.active_modules = active_modules
        self.defaults = base_optimizer.defaults
        self.ds_zero3_enabled = is_deepspeed_zero3_enabled()

        self.param_groups = base_optimizer.param_groups
        self.state_dict = base_optimizer.state_dict # for compatibility of hf Trainer

        if start_block is not None:
            self.current_block_idx = start_block
        elif self.switch_mode == "descending":
            self.current_block_idx = self.block_num - 1
        elif self.switch_mode == "ascending":
            self.current_block_idx = 0
        elif self.switch_mode == "random":
            self.block_order = torch.randperm(self.block_num).tolist()
            print_rank_0("next block epoch's update order:", self.block_order[::-1])
            self.current_block_idx = self.block_order.pop()

        # detect if in lora mode or not
        self.lora_mode = False
        if any("lora" in n for n, _ in named_parameters_list):
            self.lora_mode = True
            print_rank_0("LoRA mode detected. Will only train the lora parameters.")
            
        fp32_params = []
        for n, p in named_parameters_list:
            if p.dtype == torch.float32:
                fp32_params.append(n)
        if len(fp32_params) > 0:
            warnings.warn(f"BAdam expect model to be loaded in fp16/bf16 precision, while detect fp32"
                f"weight for the following parameters: {fp32_params} \n"
                "This will cause additional memory usage and lose the benefit of mixed precision training.")
            
        super().__init__(self.param_groups, base_optimizer.defaults)
        
        if BACKWARD_VERBOSE:
            self.record_mark = True
            self.ordered_named_params = []
            self.param_num = len(named_parameters_list)
            for n, p in named_parameters_list:
                p.register_post_accumulate_grad_hook(self.test_hook(n))

        self.switch_trainable_params()

        if BACKWARD_VERBOSE == 2:
            for name, param in self.named_parameters_list:
                param.requires_grad_(True)
    
    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p
    
    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p

    def infer_param_groups(self, param_names, include_embedding, include_lm_head):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others
        """
        
        block_prefix_list = []
        lm_head_and_other_params = []
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'

        for name in param_names:
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue
            
            if re.findall(layer_pattern, name):
                block_prefix_list.append(re.findall(layer_pattern, name))
            elif re.findall(embed_pattern, name) and include_embedding:
                block_prefix_list.append(re.findall(embed_pattern, name))
            else:
                lm_head_and_other_params.append(name)
        
        if include_lm_head:
            block_prefix_list.append(lm_head_and_other_params)
        
        return block_prefix_list
                
    def test_hook(self, name):
        """hook used for recording the time of gradient calculation, see comments on BACKWARD_VERBOSE for more details."""
        
        def func(x):
            if self.record_mark:
                self.backward_start_time = time.time()          
                self.record_mark = False
                relative_time = 0.
            else:
                relative_time = time.time() - self.backward_start_time
            if any(p_name in name for p_name in self.active_param_prefixs):
                print_rank_0(f"param: {name:<50} relative time: {relative_time}")
            
            iterator = self.named_parameters_list
                
            for n, p in iterator:
                
                if p.requires_grad and p.grad is not None:
                    print_rank_0("parameter name: ", n, "relative time", time.time() - self.backward_start_time)
                    
                    if (not any(p_name in n for p_name in self.active_param_prefixs)) and \
                        BACKWARD_VERBOSE == 2:
                        p.grad = None
                    
                    if len(self.ordered_named_params) < self.param_num:
                        self.ordered_named_params.append((n, p))
                    # break since for each step only one parameter's grad is updated
                    break
            return x
        
        return func

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        return self.base_optimizer.load_state_dict(state_dict)
    
    def _update_lr(self):
        # Make sure the learning rate of the base_optimizer is consistent with the BlockOptimizer
        for group in self.base_optimizer.param_groups:
            group["lr"] = self.param_groups[0]["lr"]

    def step(self, *args, **kwargs) -> None:
        if self.ds_zero3_enabled:
            self.step_ds_zero3(*args, **kwargs)
        else:
            self.step_single_gpu(*args, **kwargs)

        torch.cuda.empty_cache()

        if (self.global_step + 1) % self.switch_block_every == 0:
            self.switch_trainable_params()

    def step_single_gpu(self, *args, **kwargs) -> None:
        self.record_mark = True

        self._update_lr()
        self._grad_to_hp()
        self.base_optimizer.step(*args, **kwargs)
        self._update_param()
        self._clean_hp_grad()

        self.global_step += 1

    def step_ds_zero3(self, *args, **kwargs) -> None:
        """
        Basic flow: 
        1. DeepSpeedZeroOptimizer_Stage3._optimizer_step()
          * convert wrapped optim (the BlockOptimizer)'s param into hp
          * call wrapped optim's step(), i.e. this function
        2. DeepSpeedZeroOptimizer_Stage3._reassign_or_swap_out_partitioned_parameters()
          * copy hp param to lp
          * repartition the params across different GPUs
          
        In other words, deepspeed has handled the mixed-precision training, so only ordinary step is needed
        """

        self.record_mark = True

        # Since ds ZeRO-3 update the parameter in group-wise manner, 
        # we need to update the referenced of base optimizer before every step
        for i in range(len(self.param_groups)):
            self.base_optimizer.param_groups[i]["params"] = self.param_groups[i]["params"]

        self._update_lr()
        self.base_optimizer.step(*args, **kwargs)

        # ds ZeRO-3 will call step function once for each partitioned group
        self.global_step += 1/len(self.param_groups)

    def _clean_hp_grad(self) -> None:
        """Clean the gradients of the high precision parameters."""
        for hp_param in self.param_idx2hp.values():
            hp_param.grad = None

    def _update_param(self) -> None:
        """Update the low precision parameters with the values of the high precision parameters."""
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            lp_param.data.copy_(hp_param.to(lp_param.dtype).data)

    def _grad_to_hp(self, clear_lp_grads: bool = True) -> None:
        """
        Convert the gradients of the low precision parameters to high precision and calculate the gradient norm.

        Args:
            clear_lp_grads (bool, optional): Whether to clear the gradients of the low precision parameters. Defaults to True.
        """
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()

            if clear_lp_grads:
                lp_param.grad = None

    def _reset_ds_optimizer(self, trainable_param_groups):
        ds_optimizer = self.ds_optimizer
        
        # reset the bookkeeping of ds optimizer
        ds_optimizer.fp16_groups = []
        ds_optimizer.fp16_partitioned_groups = []
        ds_optimizer.fp16_partitioned_groups_flat = []
        ds_optimizer.fp16_partitioned_groups_flat_numel = []
        ds_optimizer.fp16_partitioned_groups_flat_id = []
        ds_optimizer.groups_padding = []
        ds_optimizer.fp32_partitioned_groups_flat = []
        
        # setup the fp16 groups and partition it
        ds_optimizer._create_fp16_partitions_with_defragmentation(trainable_param_groups)
        
        # register necessary hooks for synchronizing gradients
        self._create_reduce_and_remove_grad_hooks(trainable_param_groups)

        # create fp32 flat partition, initialize ipg buffer and grad partition buffer
        ds_optimizer._setup_for_real_optimizer()
        
        # invalidate the trace cache, since the module processing order has been changed
        ds_optimizer.parameter_offload.get_param_coordinator(training=True)._invalidate_trace()
        
        torch.cuda.empty_cache()

    def _create_reduce_and_remove_grad_hooks(self, trainable_param_groups):
        assert hasattr(self, "ds_optimizer"), "The optimizer doesn't have reference to its parent deepspeed optimizer yet. Set optimizer.ds_optimizer = optimizer after deepspeed.initiallize(..., optimizer=optimizer, ...)."
        ds_optimizer = self.ds_optimizer
        
        ds_optimizer.grad_accs = []
        ds_optimizer.leaf_parameters = defaultdict(list)
        for i, param_group in enumerate(ds_optimizer.fp16_groups):
            for param in param_group:
                if param.requires_grad:

                    # The hook must be created in un-partitioned parameter
                    param.all_gather()

                    def wrapper(param):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        # @instrument_w_nvtx
                        def reduce_partition_and_remove_grads(*notneeded):
                            ds_optimizer.reduce_ready_partitions_and_remove_grads(param)

                        ds_optimizer._grad_acc_hooks.append(grad_acc.register_hook(reduce_partition_and_remove_grads))
                        ds_optimizer.grad_accs.append(grad_acc)

                    #print(f"param grad fn {param.expand_as(param).grad_fn}")
                    if z3_leaf_parameter(param):
                        ds_optimizer.leaf_parameters[param.ds_z3_leaf_module].append(param)
                    else:
                        wrapper(param)

                    # Partition the parameter after creating the hook
                    param.partition()

        # We delay reduce-scatter for all gradients in the leaf modules until the backward pass of the leaf module is done
        for leaf_module, leaf_parameters in ds_optimizer.leaf_parameters.items():

            def wrapper_pre_hook(params):

                def forward_pre_hook(module, input):
                    """Pre-forward hook to set backward hook on input tensors to the leaf module"""
                    module._leaf_module_inputs_remaining = 0

                    # @instrument_w_nvtx
                    def reduce_leaf_module_grads(grad):
                        module._leaf_module_inputs_remaining -= 1
                        # Make sure everything is done in the leaf module
                        if module._leaf_module_inputs_remaining == 0:
                            for param in params:
                                if param.grad is None:
                                    param.grad = torch.zeros_like(param)
                                ds_optimizer.reduce_ready_partitions_and_remove_grads(param)

                    def set_module_bwd_hook(tensor):
                        if tensor.requires_grad:
                            module._leaf_module_inputs_remaining += 1
                            tensor.register_hook(reduce_leaf_module_grads)
                        return tensor

                    output = apply_to_tensors_only(set_module_bwd_hook, input)

                    return output

                return forward_pre_hook

            def wrapper_post_hook():

                def forward_post_hook(module, input, output):
                    """Pre-forward hook to set backward hook on input tensors to the leaf module"""
                    module._leaf_output_required_grad_num = 0

                    def increment_rg_count_bwd_hook(tensor):
                        if tensor.requires_grad:
                            module._leaf_output_required_grad_num += 1
                        return tensor

                    apply_to_tensors_only(increment_rg_count_bwd_hook, output)

                    if module._leaf_module_inputs_remaining == 0 and module._leaf_output_required_grad_num > 0:
                        raise RuntimeError(
                            "A module cannot be set as a leaf module when it does not have any input tensors that require gradients and has output tensors that require gradients. This is because the gradient reduction hook will not be called in this case."
                        )

                return forward_post_hook

            ds_optimizer._leaf_module_hooks.append(leaf_module.register_forward_pre_hook(wrapper_pre_hook(leaf_parameters)))
            ds_optimizer._leaf_module_hooks.append(leaf_module.register_forward_hook(wrapper_post_hook()))


    def switch_trainable_params(self, verbose: Optional[int] = 1) -> None:
        """
        Update the trainable parameters based on the current block index and the specified verbosity level.

        Args:
            verbose (Optional[int], optional): The verbosity level for printing information. Defaults to None.
        """
        if verbose is None:
            verbose = self.verbose

        self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules
        
        # Make sure there are trainable parameters in the current block when using lora
        while self.lora_mode:
            active_param_names = [n for n, _ in self.named_parameters_list if any(p in n for p in self.active_param_prefixs)]
            if all("lora" not in n for n in active_param_names):
                print_rank_0(f"In LoRA mode but no LoRA parameters in the current block with prefix: {self.active_param_prefixs}. Switching to the next block.")
                self._update_active_block_idx()
                self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules
                continue
            break
        
        if verbose >= 1:
            print_rank_0(f"Parameters with the following prefix will be trainable: {self.active_param_prefixs}")

        if self.ds_zero3_enabled:
            self._switch_trainable_params_zero3(verbose)
            
        else:
            self._switch_trainable_params_single_gpu(verbose)
        
        # Clean the optimizer state
        self.base_optimizer.state = defaultdict(lambda: {})
        self._update_active_block_idx()
        gc.collect()

    def _switch_trainable_params_zero3(self, verbose: Optional[int] = 1) -> None:
        assert not hasattr(self, "param_idx2lp") and not hasattr(self, "param_idx2hp")        
        
        # filter the trainable params
        trainable_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]
        
        for i, (name, param) in enumerate(self.named_parameters_list):
            if not any(p in name for p in self.active_param_prefixs):
                param.requires_grad_(False)
                param.grad = None
            else:
                if self.lora_mode and "lora" not in name:
                    continue
                param.requires_grad_(True)
                
                if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                    trainable_param_groups[0]['params'].append(param)
                else:
                    trainable_param_groups[1]['params'].append(param)

        # remove the empty param groups
        trainable_param_groups[:] = [pg for pg in trainable_param_groups if len(pg["params"]) != 0]

        self.param_groups = trainable_param_groups
        self.base_optimizer.param_groups = trainable_param_groups
        
        # During the initialization, the ds_optimizer is not set yet
        if hasattr(self, "ds_optimizer"):
    
            # remove the grad sync hooks for the previous block
            for hook in self.ds_optimizer._grad_acc_hooks:
                hook.remove()
            for hook in self.ds_optimizer._leaf_module_hooks:
                hook.remove()
            self.ds_optimizer._grad_acc_hooks.clear()
            self.ds_optimizer._leaf_module_hooks.clear()
            
            # reset the bookkeeping of ds optimizer
            self._reset_ds_optimizer(trainable_param_groups)
            
    def _switch_trainable_params_single_gpu(self, verbose: Optional[int] = 1) -> None:
        self.param_idx2lp = {}
        self.param_idx2hp = {}
        
        active_param_groups = [
            {
                "params": [],
                "weight_decay": self.param_groups[0]['weight_decay'],
                **self.defaults
            },
            {
                "params": [],
                "weight_decay": 0.0,
                **self.defaults
            },
        ]

        for i, (name, param) in enumerate(self.named_parameters_list):
            if not any(p in name for p in self.active_param_prefixs):
                param.requires_grad_(False)
                param.grad = None
            else:
                if self.lora_mode and "lora" not in name:
                    continue
                param.requires_grad_(True)
                param_hp = param.clone().float().detach().to(param.device)
                param_hp.requires_grad = True
                
                self.param_idx2lp[i] = param
                self.param_idx2hp[i] = param_hp
                
                if "bias" not in name and not isinstance(param, tuple(ALL_LAYERNORM_LAYERS)):
                    active_param_groups[0]['params'].append(param_hp)
                else:
                    active_param_groups[1]['params'].append(param_hp)
                
                if verbose >= 2:
                    print_rank_0(name)
        self.base_optimizer.param_groups = active_param_groups
        
    def _update_active_block_idx(self):
        # Update the trainable block
        if self.switch_mode == "random":
            # self.current_block_idx = random.randint(0, self.block_num - 1)
            if len(self.block_order) == 0:
                self.block_order = torch.randperm(self.block_num).tolist()
                print_rank_0("Next block epoch's update order:", self.block_order[::-1])
            self.current_block_idx = self.block_order.pop()
        elif self.switch_mode == "ascending":
            self.current_block_idx = (self.current_block_idx + 1) % self.block_num
        elif self.switch_mode == "descending":
            self.current_block_idx = (self.current_block_idx - 1) % self.block_num
        elif self.switch_mode == "fixed":
            pass
            
class BlockOptimizerRatio(Optimizer):
    """
    BlockOptimizerRatio is an extension of BlockOptimizer, where each block contains a part of trainable weights
    Args:
        param_groups (list): List of parameter groups.
        named_parameters_list (list): List of named parameters.
        update_ratio (float, optional): The update ratio for sparsification. Defaults to 0.1.
        switch_every (int, optional): Number of steps before switching to new parameter groups. Defaults to 100.
        preserve_threshold (int, optional): Threshold for preserving the whole gradient when parameter is too small. Defaults to 100.
        param_update_ratios (defaultdict, optional): Dictionary of parameter update ratios for specific parameter heads. Defaults to defaultdict(lambda: None).
        mask_mode (str, optional): Choices: ("adjacent", "scatter"). "adjacent" mode selects a group of adjacent entries in the matrix, while "scatter" selects random entries in the matrix.
        keep_mask (bool, optional): Flag to keep the mask. Defaults to True.
        include_embedding (bool, optional): Flag to include the embedding layer in optimization. Defaults to False.
    """

    def __init__(self, param_groups, 
                 named_parameters_list,
                 update_ratio=0.1, 
                 verbose=1, 
                 switch_every=100, 
                 preserve_threshold=100, 
                 param_update_ratios=defaultdict(lambda: None),
                 mask_mode = "adjacent",
                 lr=1e-5,
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 optimizer_defaults=None,
                 keep_mask=True,
                 include_embedding=False,
                 include_lm_head=False
                 ):
        self.update_ratio = update_ratio
        self.verbose = verbose
        self.sparse_hook = self.sparse_update_hook()
        self.param_groups = param_groups
        self.named_parameters_list = named_parameters_list
        self.sparse_dict = defaultdict(lambda: {})
        self.switch_every = switch_every
        self.preserve_threshold = preserve_threshold
        self.global_step = 0
        self.current_block_index = 0
        self.include_embedding = include_embedding
        
        self.param_num = len(named_parameters_list)
        self.ordered_named_params = []
        
        if not include_embedding:
            self.embedding_layer.requires_grad_(False)
        if not include_lm_head:
            self.lm_head_layer.requires_grad_(False)
        
        # mask
        self.mask_mode = mask_mode
        self.keep_mask = keep_mask
        self.mask_dict = {}
        
        for n, p in named_parameters_list:
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.sparse_hook)
            self.sparse_dict[p]["offset"] = 0
            self.sparse_dict[p]["seed"] = torch.randint(0, 1000, (1,)).item() # seed for each parameter's random index generator

            for param_name_prefix in param_update_ratios.keys():
                if param_name_prefix in n:
                    self.sparse_dict[p]["update_ratio"] = param_update_ratios[param_name_prefix]
                    continue
                    
        defaults = dict(lr=lr, betas=betas, eps=eps) if optimizer_defaults is None else optimizer_defaults
        super().__init__(self.param_groups, defaults)

    @property
    def embedding_layer(self):
        for n, p in self.named_parameters_list:
            if "embed" in n:
                return p
            
    @property
    def lm_head_layer(self):
        for n, p in self.named_parameters_list:
            if "lm_head" in n:
                return p
    
    def _sparse_adam(self,
                    params: List[Tensor],
                    grads: List[Tensor],
                    exp_avgs: List[Tensor],
                    exp_avg_sqs: List[Tensor],
                    state_steps: List[int],
                    *,
                    eps: float,
                    beta1: float,
                    beta2: float,
                    lr: float,
                    maximize: bool):
        """
        Functional API that performs Sparse Adam algorithm computation.

        Args:
            params (List[Tensor]): List of parameters.
            grads (List[Tensor]): List of gradients.
            exp_avgs (List[Tensor]): List of exponential moving average of gradients.
            exp_avg_sqs (List[Tensor]): List of exponential moving average of squared gradients.
            state_steps (List[int]): List of steps for each parameter group update.
            eps (float): Term added to the denominator to improve numerical stability.
            beta1 (float): Coefficient used for computing running averages of gradient.
            beta2 (float): Coefficient used for computing running averages of squared gradient.
            lr (float): Learning rate.
            maximize (bool): Flag to indicate if maximizing the objective function.

        """
        for i, param in enumerate(params):
            grad = grads[i]
            grad = grad if not maximize else -grad
            grad = grad.coalesce()  # the update is non-linear so indices must be unique
            grad_indices = grad._indices()
            grad_values = grad._values()
            size = grad.size()

            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            def make_sparse(values):
                constructor = grad.new
                if grad_indices.dim() == 0 or values.dim() == 0:
                    return constructor().resize_as_(grad)
                return constructor(grad_indices, values, size)

            # Decay the first and second moment running average coefficient
            #      old <- b * old + (1 - b) * new
            # <==> old += (1 - b) * (new - old)
            exp_avg_update = grad_values.sub(exp_avg).mul_(1 - beta1)
            exp_avg.add_(exp_avg_update)
            exp_avg_sq_update = grad_values.pow(2).sub_(exp_avg_sq).mul_(1 - beta2)
            exp_avg_sq.add_(exp_avg_sq_update)

            # Dense addition again is intended, avoiding another sparse_mask
            numer = exp_avg.clone()
            denom = exp_avg_sq.clone().sqrt_().add_(eps)
            del exp_avg_update, exp_avg_sq_update

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            update_direction = make_sparse(numer.div_(denom))
            param.add_(-step_size * update_direction)
            
            del update_direction

    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single AdamW optimization step, adjusted for BAdam Optimizer

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            maximize = group.get('maximize', False)

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if not p.grad.is_sparse:
                        raise RuntimeError('SparseAdam does not support dense gradients, please consider Adam instead')
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.grad._values()) # NOTE: now the exp_avg is a vector instead of matrix, since we only store the states for the non-zero entries
                        
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p.grad._values())

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            self._sparse_adam(params_with_grad,
                          grads,
                          exp_avgs,
                          exp_avg_sqs,
                          state_steps,
                          beta1=beta1,
                          beta2=beta2,
                          lr=group['lr'],
                          eps=group['eps'],
                          maximize=maximize)

        self.global_step += 1
        torch.cuda.empty_cache()
        
        if self.global_step % self.switch_every == 0:
            self._reset_state_dict()

        return loss
    
    def _reset_state_dict(self):
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = defaultdict()
    
    def _generate_mask_adjacent(self, param, ratio, offset):
        """select a group of adjacent entries in the matrix, starting from the offset. If the end of the matrix is reached, continue from the beginning."""
        num_elements = param.numel()
        num_ones = int(num_elements * ratio)
        
        if offset + num_ones > num_elements:
            i1 = torch.arange(0, offset + num_ones - num_elements, device=param.device).unsqueeze(0)
            i2 = torch.arange(offset, num_elements, device=param.device).unsqueeze(0)
            i = torch.cat([i1, i2], dim=1)
        else:
            i = torch.arange(offset, min(offset + num_ones, num_elements), device=param.device).unsqueeze(0)
        unraveled_i = torch.vstack(torch.unravel_index(i, param.size()))
        mask = torch.sparse_coo_tensor(unraveled_i, torch.ones(num_ones, device=param.device, dtype=param.dtype), param.shape)
        
        return mask
    
    def _generate_mask_scatter(self, param, ratio, offset):
        """randomly select entries in the matrix. The selected entries are not necessarily adjacent.
        The indices are recorded by setting the seed.
        """
        num_elements = param.numel()
        num_ones = int(num_elements * ratio)
        
        torch.random.manual_seed(self.sparse_dict[param]["seed"]) # NOTE: comment this seems to provide faster convergence.
        randperm = torch.randperm(num_elements, device=param.device)
        if offset + num_ones > num_elements:
            i1 = randperm[offset:]
            i2 = randperm[:offset + num_ones - num_elements]
            i = torch.cat([i1, i2])
        else:
            i = randperm[offset:offset+num_ones]
        
        unraveled_i = torch.vstack(torch.unravel_index(i, param.size()))
        mask = torch.sparse_coo_tensor(unraveled_i, torch.ones(num_ones, device=param.device, dtype=param.dtype), param.shape)
        
        return mask
    
    def sparse_update_hook(self):

        def func(p):

            num_elements = p.numel()
            offset = self.sparse_dict[p]["offset"]
            update_ratio = self.sparse_dict[p]["update_ratio"] if "update_ratio" in self.sparse_dict[p] else self.update_ratio
            
            # when the parameter is too small, we simply sparsify the whole gradient
            if num_elements < self.preserve_threshold:
                p.grad = p.grad.add_(1e-9).to_sparse()
            
            if update_ratio == 1.: # TODO: temporary inefficient fix, need to make a sparse mask
                p.grad = p.grad.add_(1e-9).to_sparse()
            else:
                if p.shape in self.mask_dict and self.mask_dict[p.shape] is not None:
                    mask = self.mask_dict[p.shape]
                else:
                    if self.mask_mode == "adjacent":
                        mask = self._generate_mask_adjacent(p, update_ratio, offset)
                    elif self.mask_mode == "scatter":
                        mask = self._generate_mask_scatter(p, update_ratio, offset)
                    else:
                        raise NotImplementedError
                    
                    # We save the same mask for all the parameters with the same shape, this treats memory for time.
                    if self.keep_mask:
                        self.mask_dict[p.shape] = mask
                
                p.grad = p.grad.sparse_mask(mask)
                    
                if (self.global_step + 1) % self.switch_every == 0:
                    self.sparse_dict[p]["offset"] = (offset + int(num_elements * update_ratio)) % num_elements
                    self.mask_dict[p.shape] = None
            
            # return p
        
        return func