#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._tsrex import TsrEx, EinopEx


def minplus(lhs: TsrEx, rhs: TsrEx):
    return EinopEx(P.ein(
        lhs._as_node(lhs), rhs._as_node(rhs), 6, 'min', 'add', None
    ))
