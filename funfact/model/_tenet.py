#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang._tsrex import TsrEx


class TensorNetwork:
    def __init__(self):
        pass

    def add_node(self, *tensor_spec_args, **tensor_spec_kwargs):
        pass

    def add_edge(self, e: TsrEx):
        pass

    def contract(self, edgelist=None):
        pass

    @property
    def nodes(self):
        pass

    @property
    def edges(self):
        pass
