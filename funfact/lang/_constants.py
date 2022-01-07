#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._ast import Primitives as P
from ._terminal import LiteralValue
from ._tsrex import TsrEx


pi = TsrEx(P.literal(LiteralValue(3.141592653589793238462643383279, r'\pi')))
