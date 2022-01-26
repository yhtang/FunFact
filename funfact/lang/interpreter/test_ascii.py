#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from unittest.mock import MagicMock as M
from .._ast import Primitives as P
from ._ascii import ASCIIRenderer


@pytest.fixture
def intr():
    return ASCIIRenderer()


@pytest.mark.parametrize('test_case', [
    (P.abstract_index_notation('', M(ascii='indices')),
     '[indices]'),

    (P.abstract_binary('', '', '', "operator"),
     'operator'),

    (P.literal('value'),
     'value'),

    (P.tensor(M(symbol='symbol')),
     'symbol'),

    (P.index(M(symbol='symbol'), False, False),
     'symbol'),

    (P.index(M(symbol='symbol'), True, False),
     '~symbol'),

    (P.index(M(symbol='symbol'), False, True),
     '*symbol'),

    # (,
    #  ),
    # (,
    #  ),
])
def test_intr(test_case, intr):
    node, result = test_case
    assert intr(node).ascii == result
