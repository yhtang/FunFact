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
        return torch.tensor(array, requires_grad=optimizable, **kwargs)

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

    @classmethod
    def sum(cls, a, axis):
        if not axis:
            return a
        else:
            return torch.sum(a, axis)

    class AutoGradMixin():
        def __iter__(self):
            for f in self.factors:
                yield f
