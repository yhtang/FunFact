#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pycuda.driver as cuda


class MetaContextManager(type):

    @property
    def context(self):
        try:
            return self._cuda_context
        except AttributeError:
            self.autoinit()
            return self._cuda_context

    def __del__(self):
        self.context.detach()


class context_manager(metaclass=MetaContextManager):

    @classmethod
    def autoinit(cls):
        if not hasattr(cls, '_cuda_context'):
            cls.init(device='auto')

    @classmethod
    def init(cls, device='auto'):
        if device == 'auto':
            cls._cuda_context = cuda.Device(0).make_context()
        else:
            cls._cuda_context = cuda.Device(device).make_context()
