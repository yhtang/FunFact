#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .rbf_expansion import RBFExpansion
from .rbf_expansion_minibatch import RBFExpansionMiniBatch


RBFExpansionV2 = RBFExpansion

__all__ = ['RBFExpansion', 'RBFExpansionV2', 'RBFExpansionMiniBatch']
