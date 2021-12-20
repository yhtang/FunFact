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
    def tensor(cls, array, optimizable, **kwargs):
        return torch.tensor(array, requires_grad=optimizable, **kwargs)

    @classmethod
    def seed(cls, key):
        cls._gen.manual_seed(key)

    @classmethod
    def normal(cls, mean, std, *shape, optimizable, dtype=torch.float32):
        return torch.normal(mean, std, shape, dtype=dtype,
                            generator=cls._gen).clone().detach().\
                                requires_grad_(optimizable)

    @classmethod
    def transpose(cls, a, axes):
        '''torch equivalent is torch.permute'''
        return torch.permute(a, (*axes,))
