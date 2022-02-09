#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from unittest.mock import MagicMock as M
from .._ast import Primitives as P
from ._slicing_propagation import SlicingPropagator


_colon = slice(None)


class T:
    pass


def test_simple():

    intr = SlicingPropagator('slices')

    with pytest.raises(NotImplementedError):
        node = intr(P.abstract_index_notation(M(), M()))

    with pytest.raises(NotImplementedError):
        node = intr(P.abstract_binary(M(), M(), M(), 'operator'))

    node = intr(P.literal(T()))
    assert not hasattr(node.value, 'slices')

    node = intr(P.tensor(T()))
    assert not hasattr(node.decl, 'slices')

    node = intr(P.index(T(), False, False))
    assert not hasattr(node.item, 'slices')

    node = intr(P.indices([T(), T()]))
    assert not hasattr(node.items[0], 'slices')
    assert not hasattr(node.items[1], 'slices')

    node = intr(P.indexed_tensor(T(), T()))
    assert node.tensor.slices == 'slices'
    assert not hasattr(node.indices, 'slices')

    node = intr(P.call('fun', M()))
    assert node.x.slices == 'slices'

    node = intr(P.neg(M()))
    assert node.x.slices == 'slices'

    node = intr(P.elem(M(), M(), 0, ''))
    assert node.lhs.slices == 'slices'
    assert node.rhs.slices == 'slices'

    with pytest.raises(NotImplementedError):
        node = intr(P.abstract_dest(M(), M()))


@pytest.mark.parametrize('test_case', [
    (
        ['i', 'j', 'k'], ['i', 'j', 'k'], (1, 2, 3), (1, 2, 3)
    ),
    (
        ['i', 'j', 'k'], ['i', 'k', 'j'], (1, 2, 3), (1, 3, 2)
    ),
    (
        ['i', 'j', 'k'], ['k', 'j', 'i'], (1, 2, 3), (3, 2, 1)
    ),
    (
        ['i', 'j', 'k'], ['k', 'i', 'j'], (1, 2, 3), (3, 1, 2)
    ),
])
def test_tran(test_case):

    indices_src, indices_dst, slices_in, slices_out = test_case

    intr = SlicingPropagator(slices_in)

    node = intr(
        P.tran(
            M(live_indices=indices_src),
            M(live_indices=indices_dst)
        )
    )

    assert node.src.slices == slices_out


@pytest.mark.parametrize('test_case', [
    (
        ['i'], ['j'], ['i', 'j'],
        (2,), (3,), (2, 3)
    ),
    (
        ['i'], ['j'], [],
        (_colon,), (_colon,), ()
    ),
    (
        ['i', 'j'], ['j'], ['i'],
        (2, _colon), (_colon,), (2,)
    ),
    (
        ['i', 'j'], ['j'], ['i'],
        (_colon, _colon), (_colon,), (_colon,)
    ),
    (
        ['i', 'j'], ['i'], ['j'],
        (_colon, 2), (_colon,), (2,)
    ),
    (
        ['i'], ['i', 'j'], ['j'],
        (_colon,), (_colon, 2), (2,)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['i', 'k'],
        (2, _colon), (_colon, 3), (2, 3)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['i', 'k'],
        (2, _colon), (_colon, _colon), (2, _colon)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['i', 'j', 'k'],
        (2, 3), (3, 4), (2, 3, 4)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['k', 'j', 'i'],
        (4, 3), (3, 2), (2, 3, 4)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['k', 'i', 'j'],
        (3, 4), (4, 2), (2, 3, 4)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['i', 'j', 'k'],
        (_colon, 3), (3, 4), (_colon, 3, 4)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['k', 'j', 'i'],
        (4, _colon), (_colon, 2), (2, _colon, 4)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['k', 'i', 'j'],
        (3, 4), (4, _colon), (_colon, 3, 4)
    ),
    (
        ['i', 'j'], ['j', 'k'], ['i', 'j', 'k'],
        (_colon, _colon), (_colon, 4), (_colon, _colon, 4)
    ),
    (
        ['i', 'j', 'k'], ['k', 'm', 'n'], ['i', 'j', 'm', 'n'],
        (2, 3, _colon), (_colon, 4, 5), (2, 3, 4, 5)
    ),
    (
        ['i', 'j', 'k'], ['m', 'n', 'k'], ['i', 'j', 'm', 'n'],
        (2, 3, _colon), (4, 5, _colon), (2, 3, 4, 5)
    ),
    (
        ['i', 'j', 'k'], ['n', 'm', 'k'], ['i', 'j', 'm', 'n'],
        (2, 3, _colon), (5, 4, _colon), (2, 3, 4, 5)
    ),
    (
        ['i', 'j', 'k'], ['k', 'm', 'j'], ['i', 'm'],
        (2, _colon, _colon), (_colon, 3, _colon), (2, 3)
    ),
])
def test_ein(test_case):

    (
        indices_lhs, indices_rhs, live_indices,
        slices_lhs, slices_rhs, slices_in
    ) = test_case

    intr = SlicingPropagator(slices_in)

    node = P.ein(
        M(live_indices=indices_lhs),
        M(live_indices=indices_rhs),
        0, '', '', None,
    )
    node.live_indices = live_indices

    node = intr(node)

    assert node.lhs.slices == slices_lhs
    assert node.rhs.slices == slices_rhs
