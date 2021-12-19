#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.backend import active_backend as ab
from ._base import TranscribeInterpreter


class LeafInitializer(TranscribeInterpreter):
    '''Creates numeric tensors for the leaf nodes in an AST.'''

    def __init__(self):
        super().__init__()

    as_payload = TranscribeInterpreter.as_payload('data')

    @as_payload
    def _wildcard(self, **kwargs):
        return None

    @as_payload
    def literal(self, value, **kwargs):
        # TODO: create tensor from literal
        return None

    @as_payload
    def tensor(self, abstract, **kwargs):
        if abstract.initializer is not None:
            if not callable(abstract.initializer):
                init_val = ab.tensor(abstract.initializer)
            else:
                init_val = abstract.initializer(abstract.shape)
        else:
            init_val = ab.normal(0.0, 1.0, *abstract.shape)
        return init_val
