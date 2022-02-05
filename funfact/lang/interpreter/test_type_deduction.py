#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from unittest.mock import MagicMock as M
from .._ast import Primitives as P
from ._type_deduction import TypeDeducer, _add_attr


@pytest.fixture
def intr():
    return TypeDeducer()


def as_payload(live_indices, keep_indices, kron_indices, shape):
    return dict(
        live_indices=live_indices,
        keep_indices=keep_indices,
        kron_indices=kron_indices,
        shape=shape
    )


@pytest.mark.parametrize('test_case', [
    (P.literal('value'),
     as_payload(None, None, None, ())),

    (P.tensor(M(shape='shape')),
     as_payload(None, None, None, 'shape')),

    (P.index('item', False, False),
     as_payload(['item'], [], [], None)),

    (P.index('item', True, False),
     as_payload(['item'], ['item'], [], None)),

    (P.index('item', False, True),
     as_payload(['item'], ['item'], ['item'], None)),

    (P.indices([
        M(**as_payload(['i'], ['j'], ['k'], None)),
        M(**as_payload(['p'], ['q'], ['r'], None)),
     ]),
     as_payload(['i', 'p'], ['j', 'q'], ['k', 'r'], None)),

    (P.indexed_tensor(
        M(shape='shape'),
        M(**as_payload(['i'], ['j'], ['k'], None))
     ),
     as_payload(['i'], ['j'], ['k'], 'shape')),

    (P.call('fun', M(**as_payload(['i'], ['j'], ['k'], 'shape'))),
     as_payload(['i'], ['j'], ['k'], 'shape')),

    (P.neg(M(**as_payload(['i'], ['j'], ['k'], 'shape'))),
     as_payload(['i'], ['j'], ['k'], 'shape')),

    (P.elem(
        M(shape=(2, 3)),
        M(shape=(2, 3)),
        '', ''),
     as_payload(None, None, None, (2, 3))),

    (P.elem(
        M(shape=(2, 3)),
        M(shape=(2, 1)),
        '', ''),
     as_payload(None, None, None, (2, 3))),

    (P.elem(
        M(shape=(1, 3)),
        M(shape=(2, 3)),
        '', ''),
     as_payload(None, None, None, (2, 3))),

    (P.elem(
        M(shape=(1, 10)),
        M(shape=(3, 1)),
        '', ''),
     as_payload(None, None, None, (3, 10))),

    (P.elem(
        M(shape=(2, 3, 1)),
        M(shape=(1, 3, 5)),
        '', ''),
     as_payload(None, None, None, (2, 3, 5))),

    (P.elem(
        M(shape=(2, 3)),
        M(shape=(2, 4)),
        '', ''),
     SyntaxError),

    (P.elem(
        M(shape=(2, 3)),
        M(shape=(2, 3, 1)),
        '', ''),
     SyntaxError),

    # i,i
    (P.ein(
        M(**as_payload(['i'], None, None, (5,))),
        M(**as_payload(['i'], None, None, (5,))),
        '', '', '',
        None
     ),
     as_payload([], [], [], ())),

    # i,j
    (P.ein(
        M(**as_payload(['i'], None, None, (5,))),
        M(**as_payload(['j'], None, None, (2,))),
        '', '', '',
        None
     ),
     as_payload(['i', 'j'], [], [], (5, 2))),

    # ~i~j,ij
    (P.ein(
        M(**as_payload(['i', 'j'], ['i', 'j'], None, (1, 10))),
        M(**as_payload(['i', 'j'], None, None, (3, 1))),
        '', '', '',
        None
     ),
     as_payload(['i', 'j'], [], [], (3, 10))),

    # ij,jk
    (P.ein(
        M(**as_payload(['i', 'j'], None, None, (5, 2))),
        M(**as_payload(['j', 'k'], None, None, (2, 10))),
        '', '', '',
        None
     ),
     as_payload(['i', 'k'], [], [], (5, 10))),

    # i~j,jk
    (P.ein(
        M(**as_payload(['i', 'j'], ['j'], None, (5, 2))),
        M(**as_payload(['j', 'k'], None, None, (2, 10))),
        '', '', '',
        None
     ),
     as_payload(['i', 'j', 'k'], [], [], (5, 2, 10))),

    # ij,~jk
    (P.ein(
        M(**as_payload(['i', 'j'], None, None, (5, 2))),
        M(**as_payload(['j', 'k'], ['j'], None, (2, 10))),
        '', '', '',
        None
     ),
     as_payload(['i', 'j', 'k'], [], [], (5, 2, 10))),

    # i~j,~jk
    (P.ein(
        M(**as_payload(['i', 'j'], ['j'], None, (5, 2))),
        M(**as_payload(['j', 'k'], ['j'], None, (2, 10))),
        '', '', '',
        None
     ),
     as_payload(['i', 'j', 'k'], [], [], (5, 2, 10))),

    # ij,jk->ki
    (P.ein(
        M(**as_payload(['i', 'j'], None, None, (5, 2))),
        M(**as_payload(['j', 'k'], None, None, (2, 10))),
        '', '', '',
        M(live_indices=['k', 'i'])
     ),
     as_payload(['k', 'i'], [], [], (10, 5))),

    # ij,jk->jki
    (P.ein(
        M(**as_payload(['i', 'j'], None, None, (5, 2))),
        M(**as_payload(['j', 'k'], None, None, (2, 10))),
        '', '', '',
        M(live_indices=['j', 'k', 'i'])
     ),
     as_payload(['j', 'k', 'i'], [], [], (2, 10, 5))),

    # *i,*i
    (P.ein(
        M(**as_payload(['i'], ['i'], ['i'], (3,))),
        M(**as_payload(['i'], ['i'], ['i'], (7,))),
        '', '', '',
        None
     ),
     as_payload(['i'], [], [], (21,))),

    # *i,*i,*i
    (P.ein(
        P.ein(
            M(**as_payload(['i'], ['i'], ['i'], (2,))),
            M(**as_payload(['i'], ['i'], ['i'], (3,))),
            '', '', '',
            None
        ),
        M(**as_payload(['i'], ['i'], ['i'], (5,))),
        '', '', '',
        None
     ),
     as_payload(['i'], [], [], (30,))),

    # *i,*i,i
    (P.ein(
        P.ein(
            M(**as_payload(['i'], ['i'], ['i'], (2,))),
            M(**as_payload(['i'], ['i'], ['i'], (3,))),
            '', '', '',
            None
        ),
        M(**as_payload(['i'], [], [], (6,))),
        '', '', '',
        None
     ),
     as_payload([], [], [], ())),

    # *i,*i,i
    (P.ein(
        P.ein(
            M(**as_payload(['i'], ['i'], ['i'], (2,))),
            M(**as_payload(['i'], ['i'], ['i'], (3,))),
            '', '', '',
            None
        ),
        M(**as_payload(['i'], [], [], (7,))),
        '', '', '',
        None
     ),
     SyntaxError),

    (P.tran(
        M(
            live_indices=['i', 'j', 'k'],
            shape=(2, 3, 4)
        ),
        M(
            live_indices=['j', 'k', 'i'],
            keep_indices='keep',
            kron_indices='kron',
            ascii='indices'
        )
     ),
     as_payload(['j', 'k', 'i'], 'keep', 'kron', (3, 4, 2))),
])
def test_concrete(test_case, intr):
    node, result = test_case
    if isinstance(result, type) and issubclass(result, Exception):
        with pytest.raises(result):
            compiled = intr(node)
    else:
        compiled = intr(node)
        for key, val in result.items():
            assert getattr(compiled, key) == val


@pytest.mark.parametrize('test_case', [
    (
        P.abstract_dest(
            P.ein(
                M(**as_payload(['i', 'j'], None, None, (5, 3))),
                M(**as_payload(['j', 'k'], None, None, (3, 4))),
                '', '', '',
                None
            ),
            M(live_indices=['k', 'i'])
        ),
        M(name_='ein', live_indices=['k', 'i'], shape=(4, 5))
    ),
    (
        P.abstract_dest(
            P.ein(
                M(**as_payload(['i', 'j'], None, None, (5, 3))),
                M(**as_payload(['j', 'k'], None, None, (3, 4))),
                '', '', '',
                None
            ),
            M(live_indices=['k', 'j', 'i'])
        ),
        M(name_='ein', live_indices=['k', 'j', 'i'], shape=(4, 3, 5))
    ),
    (
        P.abstract_dest(
            P.ein(
                M(**as_payload(['i', 'j'], None, None, (5, 3))),
                M(**as_payload(['j', 'k'], None, None, (3, 5))),
                '', '', '',
                None
            ),
            M(live_indices=['k'])
        ),
        SyntaxError
    ),
    (
        P.abstract_dest(
            P.ein(
                M(**as_payload(['i', 'j'], None, None, (5, 3))),
                M(**as_payload(['j', 'k'], None, None, (3, 5))),
                '', '', '',
                None
            ),
            M(live_indices=['k', 'l'])
        ),
        SyntaxError
    ),
    (
        P.abstract_dest(
            P.ein(
                M(**as_payload(['i', 'j'], ['j'], None, (5, 3))),
                M(**as_payload(['j', 'k'], None, None, (3, 5))),
                '', '', '',
                None
            ),
            M(live_indices=['i', 'k'])
        ),
        SyntaxError
    ),
    (
        P.abstract_dest(
            _add_attr(
                P.indexed_tensor(
                    P.tensor(M(shape=(2, 3))),
                    P.indices([
                        P.index('i', False, False),
                        P.index('j', False, False)
                    ]),
                ),
                indexed=True
            ),
            P.indices([
                P.index('j', False, False),
                P.index('i', False, False)
            ])
        ),
        M(name_='tran', live_indices=['j', 'i'], shape=(3, 2))
    ),
])
def test_abstract_dest(intr, test_case):

    abstract, concrete = test_case
    if isinstance(concrete, type) and issubclass(concrete, Exception):
        with pytest.raises(concrete):
            intr(abstract)
    else:
        compiled = intr(abstract)
        assert compiled.name == concrete.name_
        assert compiled.live_indices == concrete.live_indices
        assert compiled.shape == concrete.shape


@pytest.mark.parametrize('test_case', [
    (
        P.abstract_index_notation(
            M(indexed=False, shape='shape'),
            M(**as_payload(['i'], ['j'], ['k'], None))
        ),
        'indexed_tensor',
        as_payload(['i'], ['j'], ['k'], 'shape'),
    ),
    (
        P.abstract_index_notation(
            _add_attr(
                P.indexed_tensor(
                    P.tensor(M(shape='shape')),
                    P.indices([P.index('i', False, False)]),
                ),
                indexed=True
            ),
            P.indices([P.index('j', False, False)])
        ),
        'indexed_tensor',
        as_payload(['j'], [], [], 'shape'),
    ),
    (
        P.abstract_index_notation(
            _add_attr(
                P.indexed_tensor(
                    P.tensor(M(shape='shape')),
                    P.indices([P.index('i', False, False)]),
                ),
                indexed=True
            ),
            P.indices([P.index('j', False, False), P.index('k', False, False)])
        ),
        '',
        SyntaxError,
    ),
    (
        P.abstract_index_notation(
            _add_attr(
                P.indexed_tensor(
                    P.tensor(M(shape='shape')),
                    P.indices([
                        P.index('i', False, False),
                        P.index('j', False, False)
                    ]),
                ),
                indexed=True
            ),
            P.indices([
                P.index('j', False, False),
                P.index('k', False, False)
            ])
        ),
        'indexed_tensor',
        as_payload(['j', 'k'], [], [], 'shape'),
    ),
])
def test_abstract_index_notation(intr, test_case):

    abstract, cname, payload = test_case
    if isinstance(payload, type) and issubclass(payload, Exception):
        with pytest.raises(payload):
            intr(abstract)
    else:
        compiled = intr(abstract)
        assert compiled.name == cname
        for key, val in payload.items():
            assert getattr(compiled, key) == val


@pytest.mark.parametrize('test_case', [
    # matmul
    (
        _add_attr(
            P.abstract_binary(
                M(indexed=False, shape=(2, 3)),
                M(indexed=False, shape=(3, 4)),
                '', 'matmul'
            ),
            indexed=None
        ),
        'ein',
        (2, 4),
    ),
    (
        _add_attr(
            P.abstract_binary(
                M(indexed=False, shape=(2, 2)),
                M(indexed=False, shape=(3, 4)),
                '', 'matmul'
            ),
            indexed=None
        ),
        SyntaxError
    ),
    # kron
    (
        _add_attr(
            P.abstract_binary(
                M(indexed=False, shape=(3,)),
                M(indexed=False, shape=(5,)),
                '', 'kron'
            ),
            indexed=None
        ),
        'ein',
        (15,),
    ),
    (
        _add_attr(
            P.abstract_binary(
                M(indexed=False, shape=(2, 3)),
                M(indexed=False, shape=(5, 7)),
                '', 'kron'
            ),
            indexed=None
        ),
        'ein',
        (10, 21),
    ),
    # einop
    (
        _add_attr(
            P.abstract_binary(
                M(**as_payload(['i', 'j'], [], [], (2, 5))),
                M(**as_payload(['j', 'k'], [], [], (5, 7))),
                '', 'multiply'
            ),
            indexed=True
        ),
        'ein',
        (2, 7),
    ),
    # elementwise
    (
        _add_attr(
            P.abstract_binary(
                M(indexed=False, shape=(2, 5)),
                M(indexed=False, shape=(2, 5)),
                '', 'add'
            ),
            indexed=False
        ),
        'elem',
        (2, 5),
    ),
    (
        _add_attr(
            P.abstract_binary(
                M(indexed=False, shape=(2, 5, 1)),
                M(indexed=False, shape=(1, 5, 4)),
                '', 'subtract'
            ),
            indexed=False
        ),
        'elem',
        (2, 5, 4),
    ),
])
def test_abstract_binary(intr, test_case):

    try:
        abstract, cname, shape = test_case
        expect_exception = False
    except Exception:
        abstract, exception = test_case
        expect_exception = True

    if expect_exception:
        with pytest.raises(exception):
            intr(abstract)
    else:
        compiled = intr(abstract)
        assert compiled.name == cname
        assert compiled.shape == shape
