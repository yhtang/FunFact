#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from .rbf_expansion import RBFExpansion
from .rbf_expansion_plus import RBFExpansionPlus
from .rbf_expansion_minibatch import RBFExpansionMiniBatch
from .rbf_expansion_minibatch_plus import RBFExpansionMiniBatchPlus
from .rbf_expansion_pycuda import RBFExpansion

# RBFExpansionV2 = RBFExpansion

__all__ = [
    'RBFExpansion', 'RBFExpansionMiniBatch',
    'RBFExpansionPlus', 'RBFExpansionMiniBatchPlus'
]
