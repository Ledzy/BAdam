import torch
import random
from torch.optim import Optimizer
from torch import Tensor
from collections import defaultdict
from typing import List, Optional, Dict
import time
import math
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

# Optional [0, 1, 2]. 
    # 0: no print
    # 1: print the relative time whenever a parameter's grad is ready
    # 2: for debug usage only. Will set all the parameters trainable, print the grad ready time for each parameter. 
    #     In this case, all the grad except the "specified" trainable parameters will be set to None after being calculated.
BACKWARD_VERBOSE = 0

llama_block_prefix = [['model.layers.0.'], ['model.layers.1.'], ['model.layers.2.'], ['model.layers.3.'], 
                      ['model.layers.4.'], ['model.layers.5.'], ['model.layers.6.'], ['model.layers.7.'], ['model.layers.8.'], 
                      ['model.layers.9.'], ['model.layers.10.'], ['model.layers.11.'], ['model.layers.12.'], ['model.layers.13.'], 
                      ['model.layers.14.'], ['model.layers.15.'], ['model.layers.16.'], ['model.layers.17.'], ['model.layers.18.'], 
                      ['model.layers.19.'], ['model.layers.20.'], ['model.layers.21.'], ['model.layers.22.'], ['model.layers.23.'], 
                      ['model.layers.24.'], ['model.layers.25.'], ['model.layers.26.'], ['model.layers.27.'], ['model.layers.28.'], 
                      ['model.layers.29.'], ['model.layers.30.'], ['model.layers.31.']]

# llama_block_prefix = [['model.layers.0.input_layernorm','model.layers.0.self_attn','model.layers.0.post_attention_layernorm'], ['model.layers.0.mlp'], 
#                       ['model.layers.1.input_layernorm','model.layers.1.self_attn','model.layers.1.post_attention_layernorm'], ['model.layers.1.mlp'],
#                       ['model.layers.2.input_layernorm','model.layers.2.self_attn','model.layers.2.post_attention_layernorm'], ['model.layers.2.mlp'],
#                       ['model.layers.3.input_layernorm','model.layers.3.self_attn','model.layers.3.post_attention_layernorm'], ['model.layers.3.mlp'],
#                       ['model.layers.4.input_layernorm','model.layers.4.self_attn','model.layers.4.post_attention_layernorm'], ['model.layers.4.mlp'],
#                       ['model.layers.5.input_layernorm','model.layers.5.self_attn','model.layers.5.post_attention_layernorm'], ['model.layers.5.mlp'],
#                       ['model.layers.6.input_layernorm','model.layers.6.self_attn','model.layers.6.post_attention_layernorm'], ['model.layers.6.mlp'],
#                       ['model.layers.7.input_layernorm','model.layers.7.self_attn','model.layers.7.post_attention_layernorm'], ['model.layers.7.mlp'],
#                       ['model.layers.8.input_layernorm','model.layers.8.self_attn','model.layers.8.post_attention_layernorm'], ['model.layers.8.mlp'],
#                       ['model.layers.9.input_layernorm','model.layers.9.self_attn','model.layers.9.post_attention_layernorm'], ['model.layers.9.mlp'],
#                       ['model.layers.10.input_layernorm','model.layers.10.self_attn','model.layers.10.post_attention_layernorm'], ['model.layers.10.mlp'],
#                       ['model.layers.11.input_layernorm','model.layers.11.self_attn','model.layers.11.post_attention_layernorm'], ['model.layers.11.mlp'],
#                       ['model.layers.12.input_layernorm','model.layers.12.self_attn','model.layers.12.post_attention_layernorm'], ['model.layers.12.mlp'],
#                       ['model.layers.13.input_layernorm','model.layers.13.self_attn','model.layers.13.post_attention_layernorm'], ['model.layers.13.mlp'],
#                       ['model.layers.14.input_layernorm','model.layers.14.self_attn','model.layers.14.post_attention_layernorm'], ['model.layers.14.mlp'],
#                       ['model.layers.15.input_layernorm','model.layers.15.self_attn','model.layers.15.post_attention_layernorm'], ['model.layers.15.mlp'],
#                       ['model.layers.16.input_layernorm','model.layers.16.self_attn','model.layers.16.post_attention_layernorm'], ['model.layers.16.mlp'],
#                       ['model.layers.17.input_layernorm','model.layers.17.self_attn','model.layers.17.post_attention_layernorm'], ['model.layers.17.mlp'],
#                       ['model.layers.18.input_layernorm','model.layers.18.self_attn','model.layers.18.post_attention_layernorm'], ['model.layers.18.mlp'],
#                       ['model.layers.19.input_layernorm','model.layers.19.self_attn','model.layers.19.post_attention_layernorm'], ['model.layers.19.mlp'],
#                       ['model.layers.20.input_layernorm','model.layers.20.self_attn','model.layers.20.post_attention_layernorm'], ['model.layers.20.mlp'],
#                       ['model.layers.21.input_layernorm','model.layers.21.self_attn','model.layers.21.post_attention_layernorm'], ['model.layers.21.mlp'],
#                       ['model.layers.22.input_layernorm','model.layers.22.self_attn','model.layers.22.post_attention_layernorm'], ['model.layers.22.mlp'],
#                       ['model.layers.23.input_layernorm','model.layers.23.self_attn','model.layers.23.post_attention_layernorm'], ['model.layers.23.mlp'],
#                       ['model.layers.24.input_layernorm','model.layers.24.self_attn','model.layers.24.post_attention_layernorm'], ['model.layers.24.mlp'],
#                       ['model.layers.25.input_layernorm','model.layers.25.self_attn','model.layers.25.post_attention_layernorm'], ['model.layers.25.mlp'],
#                       ['model.layers.26.input_layernorm','model.layers.26.self_attn','model.layers.26.post_attention_layernorm'], ['model.layers.26.mlp'],
#                       ['model.layers.27.input_layernorm','model.layers.27.self_attn','model.layers.27.post_attention_layernorm'], ['model.layers.27.mlp'],
#                       ['model.layers.28.input_layernorm','model.layers.28.self_attn','model.layers.28.post_attention_layernorm'], ['model.layers.28.mlp'],
#                       ['model.layers.29.input_layernorm','model.layers.29.self_attn','model.layers.29.post_attention_layernorm'], ['model.layers.29.mlp'],
#                       ['model.layers.30.input_layernorm','model.layers.30.self_attn','model.layers.30.post_attention_layernorm'], ['model.layers.30.mlp'],
#                       ['model.layers.31.input_layernorm','model.layers.31.self_attn','model.layers.31.post_attention_layernorm'], ['model.layers.31.mlp']]


class BlockOptimizer(Optimizer):
    """Wrap the original optimizer to update trainable parameters periodically based on a specified block list."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        named_parameters_list,
        block_prefix_list: List[str],
        switch_block_every: int = 10,
        start_block: Optional[int] = None,
        switch_mode: str = "descending",
        active_modules: List[str] = [],
        verbose: int = 1,
        log_fn = None,
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
        """
        if block_prefix_list == "llama-7b":
            block_prefix_list = llama_block_prefix
        elif block_prefix_list is None:
            block_prefix_list = self.infer_param_groups([n for n, _ in named_parameters_list])

        assert switch_mode in ["random", "descending", "ascending", "fixed"]
        assert isinstance(block_prefix_list, list)

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

        self.param_groups = base_optimizer.param_groups
        self.state_dict = base_optimizer.state_dict # for compatibility of hf Trainer
        
        if start_block is not None:
            self.current_block_idx = start_block
        elif switch_mode == "descending":
            self.current_block_idx = self.block_num - 1
        elif switch_mode == "ascending":
            self.current_block_idx = 0
        elif self.switch_mode == "random":
            self.block_order = torch.randperm(self.block_num).tolist()
            print("next block epoch's update order:", self.block_order[::-1])
            self.current_block_idx = self.block_order.pop()

        # detect if in lora mode or not
        self.lora_mode = False
        if any("lora" in n for n, _ in named_parameters_list):
            self.lora_mode = True
            orint("Lora mode detected. Will only train the lora parameters.")
            
        super().__init__(self.param_groups, base_optimizer.defaults)
        
        if BACKWARD_VERBOSE:
            self.record_mark = True
            self.ordered_named_params = []
            self.param_num = len(named_parameters_list)
            for n, p in named_parameters_list:
                p.register_post_accumulate_grad_hook(self.test_hook(n))

        self.update_trainable_params()

        if BACKWARD_VERBOSE == 2:
            for name, param in self.named_parameters_list:
                param.requires_grad_(True)
                
    def infer_param_groups(self, param_names):
        """automatic inference of the parameter groups based on the parameter names.
        divide groups into:
            * embedding
            * transformer layers
            * lm_head and others
        """
        import re
        
        block_prefix_list = []
        non_embed_layer_params = []
        
        embed_pattern = r'.*embed[^.]*\.'
        layer_pattern = r'.*layers.[^.]*\.'

        for name in param_names:
            if any(prefix[0] in name for prefix in block_prefix_list):
                continue
            
            prefix = re.findall(embed_pattern + '|' + layer_pattern, name)
            if prefix:
                block_prefix_list.append(prefix)
            else:
                non_embed_layer_params.append(name)
        
        block_prefix_list.append(non_embed_layer_params)
        
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
                print(f"param: {name:<50} relative time: {relative_time}")
            
            iterator = self.named_parameters_list
                
            for n, p in iterator:
                
                if p.requires_grad and p.grad is not None:
                    print("parameter name: ", n, "relative time", time.time() - self.backward_start_time)
                    
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
        self.record_mark = True

        self._update_lr()
        self._grad_to_hp()
        self.base_optimizer.step(*args, **kwargs)
        self._update_param()
        self._clean_hp_grad()
        self.global_step += 1

        torch.cuda.empty_cache()
        if (self.global_step + 1) % self.switch_block_every == 0:
            self.update_trainable_params()

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
        grad_norm = 0.0
        for lp_param, hp_param in zip(self.param_idx2lp.values(), self.param_idx2hp.values()):
            assert lp_param.grad is not None, "The low precision parameter's gradient is None."
            hp_param.grad = lp_param.grad.float()

            if clear_lp_grads:
                lp_param.grad = None

    def update_trainable_params(self, verbose: Optional[int] = None) -> None:
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
                print(f"In LoRA mode but no lora parameters in the current block with prefix: {self.active_param_prefixs}. Switching to the next block.")
                self._update_active_block()
                self.active_param_prefixs = self.block_prefix_list[self.current_block_idx] + self.active_modules
                continue
            break
        
        if verbose >= 1:
            print("Parameters with the following prefix will be trainable:", self.active_param_prefixs)

        # Reset parameters to be optimized
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
                    print(name)

        self.base_optimizer.param_groups = active_param_groups
        
        import gc
        gc.collect()
        # Clean the optimizer state
        self.base_optimizer.state = defaultdict(lambda: {})
        self._update_active_block()

    def _update_active_block(self):
        # Update the trainable block
        if self.switch_mode == "random":
            # self.current_block_idx = random.randint(0, self.block_num - 1)
            if len(self.block_order) == 0:
                self.block_order = torch.randperm(self.block_num).tolist()
                print("Next block epoch's update order:", self.block_order[::-1])
            self.current_block_idx = self.block_order.pop()
        elif self.switch_mode == "ascending":
            self.current_block_idx = (self.current_block_idx + 1) % self.block_num
        elif self.switch_mode == "descending":
            self.current_block_idx = (self.current_block_idx - 1) % self.block_num
        elif self.switch_mode == "fixed":
            pass
            
# In BlockOptimizerRatio, each block contains a part of trainable weights
class BlockOptimizerRatio(Optimizer):
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
                #  maximize: bool = False
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
        
        self.param_num = len(named_parameters_list)
        self.ordered_named_params = []
        
        # mask
        self.mask_mode = mask_mode
        self.keep_mask = keep_mask
        self.mask_dict = {}
        
        for n, p in named_parameters_list:
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.sparse_hook)
            self.sparse_dict[p]["offset"] = 0
            self.sparse_dict[p]["seed"] = torch.randint(0, 1000, (1,)).item() # seed for each parameter's random index generator

            for param_head in param_update_ratios.keys():
                if param_head in n:
                    self.sparse_dict[p]["update_ratio"] = param_update_ratios[param_head]
                    continue
                    
        defaults = dict(lr=lr, betas=betas, eps=eps) if optimizer_defaults is None else optimizer_defaults
        super().__init__(self.param_groups, defaults)
    
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
        r"""Functional API that performs Sparse Adam algorithm computation.

        See :class:`~torch.optim.SparseAdam` for details.
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
        print("switch to new parameter groups, set the state dictionary to be zero")
    
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

        def func(x):
            if len(self.ordered_named_params) < self.param_num:
                iterator = self.named_parameters_list
            else:
                iterator = self.ordered_named_params[self.current_block_index:]
            
            for n, p in iterator:
                
                # TODO: Need to adjust the update behavior for the embedding layer
                if p.requires_grad and p.grad is not None:
                    if p.grad.is_sparse:
                        continue
                    num_elements = p.numel()
                    offset = self.sparse_dict[p]["offset"]
                    update_ratio = self.sparse_dict[p]["update_ratio"] if "update_ratio" in self.sparse_dict[p] else self.update_ratio
                    
                    # when the parameter is too small, we simply sparsify the whole gradient
                    if num_elements < self.preserve_threshold:
                        p.grad = p.grad.add_(1e-9).to_sparse()
                        if len(self.ordered_named_params) < self.param_num:
                            self.ordered_named_params.append((n, p))
                        self.current_block_index = (self.current_block_index + 1) % self.param_num
                        break
                    
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
                            if self.keep_mask:
                                self.mask_dict[p.shape] = mask
                        
                        p.grad = p.grad.sparse_mask(mask)
                            
                        if (self.global_step + 1) % self.switch_every == 0:
                            self.sparse_dict[p]["offset"] = (offset + int(num_elements * update_ratio)) % num_elements
                            self.mask_dict[p.shape] = None
                    
                    if len(self.ordered_named_params) < self.param_num:
                        self.ordered_named_params.append((n, p))
                    
                    self.current_block_index = (self.current_block_index + 1) % self.param_num
                    # break since for each step only one parameter's grad is updated
                    break 
                else:
                    if len(self.ordered_named_params) == self.param_num:
                        print("The parameter's update order has been changed. This may indicate bugs in sparse optimizer.")
            return x
        
        return func