#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from ._tsrex import TsrEx
from ._ast import _ASNode
from ._semiring import minplus, logsumexp, viterbi


@pytest.mark.parametrize('test_case', [
    (minplus,),
    (logsumexp,),
    (viterbi,)
])
def test_semiring_operation(test_case):

    op, = test_case
    tsrex = op(_ASNode(), _ASNode())

    assert isinstance(tsrex, TsrEx)
    assert tsrex.root.name == 'ein'
    assert tsrex.root.precedence == 6
    assert tsrex.root.outidx is None
