from transformers.trainer_callback import TrainerCallback
from transformers.utils import logging
from transformers.integrations import is_deepspeed_zero3_enabled
from typing import Union, Iterable, Any, Dict, Optional
from types import MethodType
import torch

logger = logging.get_logger(__name__)

# class BAdamZeRO3Callback(TrainerCallback):
class BAdamCallback(TrainerCallback):
    """ Handler for setup BAdam's training process with ZeRO-3. """
    def __init__(self, *args, **kwargs):
        self.init_loss_scale = kwargs.get("init_loss_scale", 12)
        
        
    def on_train_begin(self, *args, **kwargs):
        model = kwargs["model"]

        if hasattr(model, "disable_input_require_grads") and hasattr(model, "_require_grads_hook"):
            model.disable_input_require_grads()
            logger.info("Disable embedding output's require_grads for block-wise optimizer. Instead, "
                        "set input of checkpoint layer's `requires_grad` to True when the checkpoint layer is trainable")

        model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable_for_bcd, model)

        if is_deepspeed_zero3_enabled():
            optimizer = kwargs["optimizer"] # DeepSpeedOptimizerWrapper
            
            # Create the BlockOptimizer's reference to DeepSpeedZeroOptimizer_Stage3
            ds_optim = optimizer.optimizer # DeepSpeedZeroOptimizer_Stage3
            badam_optim = ds_optim.optimizer # BlockOptimizer
            badam_optim.ds_optimizer = ds_optim
        
            # adjust the loss scale when it is not specified in the configuration file
            if not hasattr(ds_optim, "dynamic_loss_args"):
                ds_optim.cur_scale = 2**self.init_loss_scale
                logger.info(f"Reducing initial loss scale to {ds_optim.cur_scale} for avoiding unnecessary attempts.")


def gradient_checkpointing_enable_for_bcd(
    self: "PreTrainedModel", gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
) -> None:
    r"""
    Modification of the original method to enable gradient checkpointing for block-wise optimizer.

    To backward Pytorch checkpoint layer, the input of the backward layer should be a tensor 
    with `requires_grad=True`. In full parameter training scheme, the output of the embedding's 
    `requires_grad` is set to True (by model.enable_input_require_grads()). However, when using 
    Block-wise update, backward to the embedding layer is not necessary and induces additional
    memory and time cost. Therefore, we disable the `requires_grad` of the embedding layer's output,
    and apply this function to the make input's `requires_grad`  to True when the checkpoint layer 
    has trainable parameters.
    """
    from torch.utils.checkpoint import checkpoint
    from functools import partial
    import inspect

    if not self.supports_gradient_checkpointing:
        raise ValueError("{} does not support gradient checkpointing.".format(self.__class__.__name__))

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    def custom_gradient_checkpointing_func(func, *args, **kwargs):
        module: "torch.nn.Module" = func.__self__

        if any(param.requires_grad for param in module.parameters()):
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)

        return gradient_checkpointing_func(func, *args, **kwargs)

    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.enable_input_require_grads()
        logger.warning("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=custom_gradient_checkpointing_func)


# For torch>=2.1, `_foreach_norm` is used when implementing `clip_grad_norm_`, which doesn't support sparse tensor yet.
# We can temporarily fix this issue by using the older torch version's implementation:
    # self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)
def clip_grad_norm_old_version(self, parameters, max_norm, norm_type=2):
    """
    Modification of the accelerator.clip_grad_norm_ to enable gradient clipping for sparse tensor.
    This is necessary when using BlockOptimizerRatio. 
    add the following line at the end of your Trainer's __init__ if gradient clip is applied and your torch version >= 2.1:
    `self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)` 
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