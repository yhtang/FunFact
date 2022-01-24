#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from ._tsrex import TsrEx
from ._special import zeros, ones, eye


@pytest.mark.parametrize('test_case', [
    (zeros,),
    (ones,),
    (eye,)
])
def test_special_tensor(test_case):

    gen, = test_case

    for shape in [(2, 3), (3, 3), (1, 7)]:
        t = gen(*shape)

        assert isinstance(t, TsrEx)
        assert t.root.name == 'tensor'

    if gen is not eye:
        for shape in [(2, 3, 4), (1,)]:
            t = gen(*shape)

            assert isinstance(t, TsrEx)
            assert t.root.name == 'tensor'
