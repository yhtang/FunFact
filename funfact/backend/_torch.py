#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch


__name__ = 'PyTorchBackend'


nla = torch
native_t = torch.Tensor
tensor_t = (torch.Tensor, np.ndarray)

_gen = torch.Generator()


def set_context(**context):
    pass


def tensor(array, optimizable=False, **kwargs):
    if type(array) is native_t:
        t = array.clone().detach()
    else:
        t = torch.tensor(array, **kwargs)
    return set_optimizable(t, optimizable)


def to_numpy(tensor, **kwargs):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return np.asarray(tensor.numpy(), **kwargs)


def seed(key):
    _gen.manual_seed(key)


def normal(mean, std, shape, dtype=torch.float32):
    with torch.no_grad():
        return torch.normal(
            mean, std, shape, dtype=dtype, generator=_gen
        )


def uniform(low, high, shape, dtype=torch.float32):
    with torch.no_grad():
        return torch.rand(
            shape, dtype=dtype, generator=_gen
        ) * (high - low) + low


def transpose(a, axes):
    '''torch equivalent is torch.permute'''
    return torch.permute(a, (*axes,))


def sum(a, axis=None):
    if axis:
        return torch.sum(a, axis)
    else:
        return torch.sum(a)


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
