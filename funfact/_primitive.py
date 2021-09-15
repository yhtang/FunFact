#!/usr/bin/env python
# -*- coding: utf-8 -*-


precedence = dict(
    lit=0,     # literal values
    idx=1,     # tensor indexing
    call=2,    # function call
    square=3,  # element-wise square
    neg=4,     # negation
    mul=5,     # multiplication
    div=5,     # division
    add=6,     # addition
    sub=6,     # subtraction
)
