#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from unittest.mock import MagicMock as M
from .._ast import Primitives as P
from ._slicing_propagation import SlicingPropagator


@pytest.mark.parametrize('test_case', [
    (P.abstract_index_notation('', M(ascii='indices')),
     NotImplementedError),

    (P.abstract_binary('', '', '', "operator"),
     NotImplementedError),

    (P.literal(''),
     None),

    (P.tensor(M()),
     None),

    (P.indices([]),
     None),

    (P.indexed_tensor('', ''),
     'slices'),

    (P.call('fun', ''),
     'slices'),

    (P.neg(''),
     'slices'),

    (P.elem('', '', '', 'operator'),
     'slices'),

    # (P.ein('', '', '', 'red', 'pair', M(ascii='outidx')),
    #  'red:pair -> outidx'),

    # (P.ein('', '', '', 'red', 'pair', None),
    #  'red:pair'),

    # (P.tran('', M(ascii='indices')),
    #  '-> [indices]'),

    (P.abstract_dest('', M(ascii='indices')),
     NotImplementedError),
])
def test_simple(test_case):
    intr = SlicingPropagator('slices')
    node, result = test_case
    if result is NotImplementedError:
        with pytest.raises(result):
            intr(node)
    else:
        assert intr(node).slices == result


# @pytest.fixture
# def intr():
#     return SlicingPropagator()


# @pytest.mark.parametrize('test_case', [
#     (P.abstract_index_notation('', M(ascii='indices')),
#      NotImplementedError),

#     (P.abstract_binary('', '', '', "operator"),
#      NotImplementedError),

#     (P.literal('value'),
#      'value'),

#     # (P.tensor(M(symbol='symbol')),
#     #  'symbol'),

#     # (P.index(M(symbol='symbol'), False, False),
#     #  'symbol'),

#     # (P.index(M(symbol='symbol'), True, False),
#     #  '~symbol'),

#     # (P.index(M(symbol='symbol'), False, True),
#     #  '*symbol'),

#     # (P.indices([M(ascii='i'), M(ascii='j')]),
#     #  'i,j'),

#     # (P.indexed_tensor('', M(ascii='indices')),
#     #  '[indices]'),

#     # (P.call('fun', ''),
#     #  'fun'),

#     # (P.neg(''),
#     #  ''),

#     # (P.elem('', '', '', 'operator'),
#     #  'operator'),

#     # (P.ein('', '', '', 'red', 'pair', M(ascii='outidx')),
#     #  'red:pair -> outidx'),

#     # (P.ein('', '', '', 'red', 'pair', None),
#     #  'red:pair'),

#     # (P.tran('', M(ascii='indices')),
#     #  '-> [indices]'),

#     (P.abstract_dest('', M(ascii='indices')),
#      NotImplementedError),
# ])
# def test_intr(test_case, intr):
#     node, result = test_case
#     if result is NotImplementedError:
#         with pytest.raises(result):
#             intr(node)
#     # assert intr(node).slices == result
