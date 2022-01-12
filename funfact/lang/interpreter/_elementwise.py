#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter._evaluation import Evaluator


class ElementwiseEvaluator(Evaluator):
    def tensor(self, decl, data, slices, **kwargs):
        return data[tuple(slices)]
