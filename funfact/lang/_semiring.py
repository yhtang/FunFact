#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._tsrex import TsrEx, EinopEx, _BaseEx


def minplus(lhs: TsrEx, rhs: TsrEx):
    return EinopEx(P.ein(
        _BaseEx(lhs).root, _BaseEx(rhs).root, 6, 'min', 'add', None
    ))


def logspacesum(lhs: TsrEx, rhs: TsrEx):
    return EinopEx(P.ein(
        _BaseEx(lhs).root, _BaseEx(rhs).root, 6, 'logspace_sum', 'add', None
    ))
