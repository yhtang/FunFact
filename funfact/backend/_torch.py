#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from ._meta import BackendMeta


class PyTorchBackend(metaclass=BackendMeta):

    _nla = torch
    _gen = torch.Generator()

    native_t = torch.Tensor
    tensor_t = (torch.Tensor, np.ndarray)

    @classmethod
    def tensor(cls, array, optimizable=False, **kwargs):
        if type(array) is cls.native_t:
            return array.clone().detach().requires_grad_(optimizable)
        return torch.tensor(array, requires_grad=optimizable, **kwargs)

    @classmethod
    def to_numpy(cls, tensor):
        if tensor.requires_grad:
            return tensor.detach().numpy()
        else:
            return tensor.numpy()

    @classmethod
    def seed(cls, key):
        cls._gen.manual_seed(key)

    @classmethod
    def normal(cls, mean, std, *shape, optimizable=True, dtype=torch.float32):
        data = torch.normal(mean, std, shape, dtype=dtype, generator=cls._gen)
        if optimizable:
            return data.clone().detach().requires_grad_(True)
        else:
            return data

    @classmethod
    def transpose(cls, a, axes):
        '''torch equivalent is torch.permute'''
        return torch.permute(a, (*axes,))

    @classmethod
    def reshape(cls, a, newshape, order='C'):
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

    @staticmethod
    def loss_and_grad(loss_fn, example_model, example_target):
        def wrapper(model, target):
            loss = loss_fn(model(), target)
            gradients = torch.autograd.grad(loss, model)
            # gradients = [data.grad for data in model.factors]
            return loss, gradients
        return wrapper
        return torch.jit.trace(wrapper, (example_model, example_target))
        # TODO jit trace with Factorization class

    def autograd_decorator(ob):
        return ob

    class AutoGradMixin():
        def __iter__(self):
            for f in self.factors:
                yield f
