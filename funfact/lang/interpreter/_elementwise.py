#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact.lang.interpreter._evaluation import Evaluator


class ElementEvaluator(Evaluator):
    def tensor(self, abstract, data, slices, **kwargs):
        return data[slices]
