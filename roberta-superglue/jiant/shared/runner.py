import os
from typing import Union, Optional, Iterable

import torch
import torch.nn as nn

import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.torch_utils as torch_utils


def complex_backpropagate(
    loss, optimizer, model, fp16, n_gpu, gradient_accumulation_steps, max_grad_norm
):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    if fp16:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from apex import amp

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
    else:
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        clip_func_(model.parameters(), max_grad_norm)
    return loss

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
    if norm_type == torch.inf:
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

def get_train_dataloader_from_cache(
    train_cache: caching.ChunkedFilesDataCache, task, train_batch_size: int, explicit_subset=None
):
    # TODO: Expose buffer_size parameter  (issue #1183)
    dataset = train_cache.get_iterable_dataset(buffer_size=10000, shuffle=True, explicit_subset=explicit_subset)
    train_dataloader = torch_utils.DataLoaderWithLength(
        dataset=dataset,
        batch_size=train_batch_size,
        collate_fn=task.collate_fn,
    )
    return train_dataloader


def get_eval_dataloader_from_cache(
    eval_cache: caching.ChunkedFilesDataCache,
    task,
    eval_batch_size: int,
    subset_num=None,
    explicit_subset=None,
):
    dataset = eval_cache.get_iterable_dataset(
        buffer_size=10000,
        shuffle=False,
        subset_num=subset_num,
        explicit_subset=explicit_subset,
    )
    eval_dataloader = torch_utils.DataLoaderWithLength(
        dataset=dataset,
        batch_size=eval_batch_size,
        collate_fn=task.collate_fn,
    )
    return eval_dataloader


def save_model_with_metadata(
    model_or_state_dict: Union[nn.Module, dict],
    output_dir: str,
    file_name="model",
    metadata: Optional[dict] = None,
):
    if isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    else:
        state_dict = torch_utils.get_model_for_saving(model_or_state_dict).state_dict()

    torch.save(state_dict, os.path.join(output_dir, f"{file_name}.p"))
    if metadata is not None:
        py_io.write_json(metadata, os.path.join(output_dir, f"{file_name}.metadata.json"))


def compare_steps_max_steps(step, max_steps):
    return max_steps is not None and max_steps != -1 and step >= max_steps
