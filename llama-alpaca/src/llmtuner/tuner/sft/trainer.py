import os
import json
import torch
import numpy as np
import torch.nn as nn
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger
from llmtuner.hparams import FinetuningArguments
from llmtuner.tuner.core.trainer import PeftTrainer
from ..core.utils import set_trainable, set_trainable_block, llama_param_groups

from .block_optim import BlockOptimizer, BlockOptimizerRatio
import time
from types import MethodType
from functools import wraps
from torch.utils.checkpoint import checkpoint
from packaging import version
if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput


logger = get_logger(__name__)

def time_and_record(original_method, log_fn, prefix):
    
    @wraps(original_method) # inherit the signature of the original method
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = original_method(*args[1:], **kwargs) # skip the first argument (self)
        end_time = time.time()
        log_fn({f"train/{prefix}": end_time - start_time,
                f"after_{prefix}": end_time})
        return result
        # return MethodType(wrapper, original_method.__self__)(*args, **kwargs) # make it a class method

    return MethodType(wrapper, original_method.__self__) # make it a class method

def time_forward(original_method, log_fn, prefix): #TODO: temporary solution, avoid using MethodType to wrap forward function so that __self__ is preserved
        
    @wraps(original_method) # inherit the signature of the original method
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = original_method(*args[1:], **kwargs)
        end_time = time.time()
        log_fn({f"train/{prefix}": end_time - start_time})
        return result

    return wrapper

def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    """
    Modification of the original method to enable gradient checkpointing for block-wise optimizer.
    
    Args:
        gradient_checkpointing_kwargs (dict, *optional*):
            Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
    """
    
    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    # gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_func(func, *args, **kwargs):
        module = func.__self__
        
        # torch.is_grad_enabled is used to check whether in inference mode or not; helps accelerate inference speed.
        if torch.is_grad_enabled() and any([p.requires_grad for p in module.parameters()]):
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)
        
        return checkpoint(func, *args, **kwargs)            
        
    self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)

    if getattr(self, "_hf_peft_config_loaded", False):
        # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
        # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
        # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
        # the gradients to make sure the gradient flows.
        self.enable_input_require_grads()

def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
    """
    Modification of the accelerator.clip_grad_norm_ to enable gradient clipping for sparse tensor.
    Used for torch version >= 2.1
    """
    from accelerate.utils import DistributedType
    from torch import inf

    if self.distributed_type == DistributedType.FSDP:
        self.unscale_gradients()
        parameters = [p for p in parameters]
        for model in self._models:
            if parameters == [p for p in model.parameters()]:
                return model.clip_grad_norm_(max_norm, norm_type)
    elif self.distributed_type == DistributedType.DEEPSPEED:
        # `accelerator.backward(loss)` is doing that automatically. Therefore, its implementation is not needed
        # We cannot return the gradient norm because DeepSpeed does it.
        return None
    self.unscale_gradients()
    
    def clip_func_(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]], max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
        r""" torch 1.13 version clip_grad_norm_, works well with sparse tensor.
        Clips gradient norm of an iterable of parameters.

        The norm is computed over all gradients together, as if they were
        concatenated into a single vector. Gradients are modified in-place.

        Args:
            parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
                single Tensor that will have gradients normalized
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.
            error_if_nonfinite (bool): if True, an error is thrown if the total
                norm of the gradients from :attr:`parameters` is ``nan``,
                ``inf``, or ``-inf``. Default: False (will switch to True in the future)

        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad for p in parameters if p.grad is not None]
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if len(grads) == 0:
            return torch.tensor(0.)
        device = grads[0].device
        if norm_type == inf:
            norms = [g.detach().abs().max().to(device) for g in grads]
            total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
        else:
            total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
        if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
            raise RuntimeError(
                f'The total norm of order {norm_type} for gradients from '
                '`parameters` is non-finite, so it cannot be clipped. To disable '
                'this error and scale the gradients by the non-finite norm anyway, '
                'set `error_if_nonfinite=False`')
        clip_coef = max_norm / (total_norm + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
        # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
        # when the gradients do not reside in CPU memory.
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))
        return total_norm
    
    return clip_func_(parameters, max_norm, norm_type=norm_type)

class Seq2SeqPeftTrainer(PeftTrainer):
    r"""
    Inherits PeftTrainer to compute generative metrics such as BLEU and ROUGE.
    """
    
    def __init__(self, finetuning_args: FinetuningArguments, **kwargs):
        super().__init__(finetuning_args, **kwargs)
        self.num_layers = 32
        self.current_layer_idx = 31
        self.step_counter = 0
        self.model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, self.model)
        self.model.gradient_checkpointing_enable()
        self.optimizer = self.create_optimizer()
        
        # Time these functions
        self.model.forward = time_forward(self.model.forward, self.log, "forward_time")
        self.optimizer.step = time_and_record(self.optimizer.step, self.log, "step_time")
        self.accelerator.backward = time_and_record(self.accelerator.backward, self.log, "backward_time")
        
        if version.parse(torch.__version__) >= version.parse("1.13"):
            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_, self.accelerator)
            
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        self.accelerator.wait_for_everyone()
        self.model = self.accelerator.unwrap_model(self.model)

    def casted_step(self, *args, **kwargs):
        with torch.autocast(device_type=torch.float32):
            super().step(self, *args, **kwargs)            
    def create_optimizer(self):
        """
        Adjust the original create_optimizer method to allow sparse update.
        """
        from transformers.utils.import_utils import is_sagemaker_mp_enabled
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        
        if self.finetuning_args.finetuning_type == "sparse":
            print("Using Sparse optimizer")
            self.optimizer = BlockOptimizerRatio(param_groups=optimizer_grouped_parameters,
                                                named_parameters_list=list(self.model.named_parameters()),
                                                switch_every=self.finetuning_args.switch_block_every,
                                                update_ratio=self.finetuning_args.update_ratio,
                                                optimizer_defaults=optimizer_kwargs) # TODO: make it an argument
            return self.optimizer
        
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.finetuning_args.finetuning_type == "block":
            # make it block-wise optimizer
            print("Using block-wise optimizer")
            self.optimizer = BlockOptimizer(base_optimizer=self.optimizer, 
                                        named_parameters_list=list(self.model.named_parameters()),
                                        block_prefix_list="llama-7b",
                                        switch_block_every=self.finetuning_args.switch_block_every,
                                        switch_mode=self.finetuning_args.switch_mode,
                                        log_fn=self.log,
                                        start_block=self.finetuning_args.start_block,
                                        verbose=self.finetuning_args.badam_verbose)
        elif self.finetuning_args.finetuning_type == "lora":
            # when using LoRA, make sure the update uses float32 # TODO, fix bug
            # self.step = self.casted_step
            pass
        elif self.finetuning_args.finetuning_type != "full": #TODO: resolve the saving imcompatibility when setting finetuning_type to full
            raise ValueError(f"Invalid finetuning type: {self.finetuning_args.finetuning_type}")

        return self.optimizer

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
        if prompt_len > label_len:
            inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
        if label_len > prompt_len:
            inputs["input_ids"] = self._pad_tensors_to_target_len(inputs["input_ids"], inputs["labels"])
            if "attention_mask" in inputs:
                inputs["attention_mask"] = self._pad_tensors_to_target_len(
                    inputs["attention_mask"], inputs["labels"], pad_token_id=0
                )
            if "position_ids" in inputs:
                inputs["position_ids"] = self._pad_tensors_to_target_len(
                    inputs["position_ids"], inputs["labels"], pad_token_id=0
                )

        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None:
            generated_tokens[:, :max(prompt_len, label_len)] = (
                self.tokenizer.pad_token_id * torch.ones_like(generated_tokens[:, :max(prompt_len, label_len)])
            )

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(
        self,
        src_tensor: torch.Tensor,
        tgt_tensor: torch.Tensor,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.

        Should only be called when predict_with_generate=True.
        """
        if pad_token_id is None:
            if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
                assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
                pad_token_id = self.tokenizer.pad_token_id
            else:
                raise ValueError("PAD token is required.")

        padded_tensor = pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1]:] = src_tensor # adopt left-padding
        return padded_tensor.contiguous() # in contiguous memory

    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    def evaluate(self, *args, **kwargs):
        # train_metrics = super().evaluate(self.train_dataset)
        # self.log({"eval/train_loss": train_metrics['eval_loss']})
        metrics = super().evaluate(*args, **kwargs)
        self.log({"eval/eval_loss": metrics['eval_loss']})
        
        return metrics