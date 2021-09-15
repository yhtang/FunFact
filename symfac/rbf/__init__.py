#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from .rbf_expansion import RBFExpansion
from ._dense_fullgrad import RBFExpansionDenseFullGrad
from ._dense_stochgrad import RBFExpansionDenseStochasticGrad

# RBFExpansionV2 = RBFExpansion

__all__ = [
    'RBFExpansionDenseFullGrad', 'RBFExpansionDenseStochasticGrad'
]
