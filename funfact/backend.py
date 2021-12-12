#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np


class BackendMeta(ABCMeta):

    def __getattr__(self, attr):
        return getattr(self._np, attr)





class JAXBackend(Backend):
    pass


class PyTorchBackend(Backend):
    pass


backend = JAXBackend()

class DummyBackend:




# def use(backend: Union[str, Backend]):
#     pass
