#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from ._meta import BackendMeta


class PyTorchBackend(metaclass=BackendMeta):

    _nla = torch

    tensor_t = torch.Tensor

    @classmethod
    def as_tensor(cls, array, **kwargs):
        return torch.as_tensor(array, **kwargs)
