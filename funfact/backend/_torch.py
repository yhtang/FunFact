#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''[PyTorch](https://pytorch.org/).'''
import numpy as np
import torch
from funfact.util.iterable import as_tuple
from ._context import context


__name__ = 'PyTorchBackend'

nla = torch
native_t = torch.Tensor
'''The native type for tensor data used by the backend.'''
tensor_t = (torch.Tensor, np.ndarray)
'''Types acceptable by the backend API as 'tensors'.'''

_device = torch.device(context.get('device', 'cpu'))
_gen = torch.Generator()


def tensor(array, optimizable=False, **kwargs):
    if type(array) is native_t:
        t = array.clone().detach().to(_device)
    else:
        t = torch.tensor(array, **kwargs, device=_device)
    return set_optimizable(t, optimizable)


def to_numpy(tensor, **kwargs):
    if tensor.requires_grad:
        tensor = tensor.detach()
    if _device.type != 'cpu':
        tensor = tensor.cpu()
    return np.asarray(tensor.numpy(), **kwargs)


def _add_device(f):
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs, device=_device)
    return wrapped


zeros = _add_device(torch.zeros)
zeros_like = _add_device(torch.zeros_like)
ones = _add_device(torch.ones)
ones_like = _add_device(torch.ones_like)
arange = _add_device(torch.arange)
linspace = _add_device(torch.linspace)
logspace = _add_device(torch.logspace)
eye = _add_device(torch.eye)
empty = _add_device(torch.empty)
empty_like = _add_device(torch.empty_like)
empty_strided = _add_device(torch.empty_strided)
full = _add_device(torch.full)
full_like = _add_device(torch.full_like)
complex = _add_device(torch.complex)


def seed(key):
    _gen.manual_seed(key)


def normal(mean, std, shape, dtype=torch.float32):
    with torch.no_grad():
        return torch.normal(
            mean, std, as_tuple(shape), dtype=dtype, generator=_gen
        ).to(_device)


def uniform(low, high, shape, dtype=torch.float32):
    with torch.no_grad():
        return torch.rand(
            as_tuple(shape), dtype=dtype, generator=_gen
        ).to(_device) * (high - low) + low


def transpose(a, axes):
    '''torch equivalent is torch.permute'''
    return torch.permute(a, (*axes,))


def sum(a, axis=None):
    if axis:
        return torch.sum(a, axis)
    else:
        return torch.sum(a)


def power(input, exponent):
    return torch.pow(input, exponent)


def reshape(a, newshape, order='C'):
    if order == 'C':
        return torch.reshape(a, (*newshape,))
    elif order == 'F':
        if len(a.shape) > 0:
            a = a.permute(*reversed(range(len(a.shape))))
        return a.reshape(*reversed(newshape)).permute(
                    *reversed(range(len(newshape))))
    else:
        raise ValueError(
            f'Unsupported option for reshape order: {order}.'
        )


def loss_and_grad(loss_fn, example_model, example_target, **kwargs):
    def wrapper(model, target):
        loss = loss_fn(model, target, **kwargs)
        gradients = torch.autograd.grad(loss, model)
        # gradients = [data.grad for data in model.factors]
        return loss, gradients
    return wrapper
    return torch.jit.trace(wrapper, (example_model, example_target))
    # TODO jit trace with Factorization class


def add_autograd(cls):

    class AddAutoGrad(cls):
        def __iter__(self):
            for f in self.factors:
                yield f

    return AddAutoGrad


def set_optimizable(x: native_t, optimizable: bool):
    return x.requires_grad_(optimizable)
