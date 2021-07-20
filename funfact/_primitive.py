#!/usr/bin/env python
# -*- coding: utf-8 -*-
import typing


class Primitive(typing.NamedTuple):
    name: str
    symbol: str
    tex: str
    precedence: int


p_idx = Primitive('idx', '[]', '_', 1)
p_div = Primitive('exp', '**', '^', 3)
p_neg = Primitive('neg', '-', '-', 4)
p_mul = Primitive('mul', '*', '', 5)
p_div = Primitive('div', '/', '/', 5)
p_add = Primitive('add', '+', '+', 6)
p_sub = Primitive('sub', '-', '-', 6)
