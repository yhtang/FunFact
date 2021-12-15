#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from ._meta import BackendMeta


class PyTorchBackend(metaclass=BackendMeta):

    _nla = torch
    _gen = torch.Generator()

    tensor_t = torch.Tensor

    @classmethod
    def as_tensor(cls, array, **kwargs):
        return torch.as_tensor(array, **kwargs)

    @classmethod
    def seed(cls, key):
        cls._gen.manual_seed(key)

    @classmethod
    def normal(cls, mean, std, *shape, dtype=torch.float32):
        return torch.normal(mean, std, shape, dtype=dtype, generator=cls._gen)

    @classmethod
    def transpose(cls, a, axes):
        '''torch equivalent is torch.permute'''
        return torch.permute(a, (*axes,))
