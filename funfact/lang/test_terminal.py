#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import uuid
import multiprocessing
import numpy as np
from ._terminal import (
    Symbol,
    Identifiable,
    LiteralValue,
    AbstractIndex,
    AbstractTensor,
)


@pytest.mark.parametrize(
    'identifier', [
        ('a', '1'),
        ('a', None),
        'a_1',
        'a'
    ]
)
def test_symbol_init(identifier):
    print('identifier', identifier)
    s = Symbol(identifier)
    assert s.letter == 'a'
    assert (s.number.isdigit() if s.number is not None else True)
    assert not repr(s).startswith('<')
    assert not repr(s).endswith('>')
    assert str(s) != repr(s)
    assert s.letter in str(s)
    assert repr(s) == repr(eval(repr(s)))


@pytest.mark.parametrize(
    'invalid_identifier', [
        '123',
        '123_123',
        '123_a',
        '_1',
        '__1',
        '1_2_3',
        'a_1_1',
        'a_b_c',
        123,
        123.0,
        True,
        None
    ]
)
def test_symbol_init_exception(invalid_identifier):
    with pytest.raises(RuntimeError):
        Symbol(invalid_identifier)


def test_symbol_init_by_uuid():
    class MockSpecializedSymbol(Symbol):
        _anon_registry = {}
        _anon_registry_lock = multiprocessing.Lock()

    u = uuid.uuid4()
    v = uuid.uuid4()

    r = MockSpecializedSymbol(u)
    assert r.letter is None
    assert r.number.isdigit()

    s = MockSpecializedSymbol(v)
    assert s.letter is None
    assert s.number.isdigit()

    t = MockSpecializedSymbol(u)
    assert t.letter is None
    assert t.number == r.number


def test_identifiable():
    a = Identifiable()
    b = Identifiable()
    assert a == a
    assert b == b
    assert a != b
    assert hash(a) == hash(a)
    assert hash(b) == hash(b)
    assert hash(a) != hash(b)


def test_literal_value():
    _0_tex = r'\mathbf{0}'
    _0 = LiteralValue(0, _0_tex)
    assert isinstance(_0, Identifiable)
    assert str(0) in str(_0)
    assert repr(_0) == repr(eval(repr(_0)))
    assert _0._repr_tex_() == _0_tex


@pytest.mark.parametrize(
    'symbol', [
        'i',
        'i_1',
    ]
)
def test_abstract_index(symbol):

    i = AbstractIndex(symbol)
    assert i == i
    assert isinstance(repr(i), str)
    assert isinstance(str(i), str)
    assert isinstance(i._repr_tex_(), str)
    assert r'\bar' in i._repr_tex_(accent=r'\bar')
    assert i._repr_tex_() in i._repr_html_()


def test_abstract_index_equality():

    i = AbstractIndex()
    j = AbstractIndex()
    k = AbstractIndex('k_1')
    assert i != j
    assert i != k
    assert j != k

    assert hash(i) != hash(j)
    assert hash(i) != hash(k)
    assert hash(j) != hash(k)


def test_abstract_tensor():

    u = AbstractTensor(3)
    v = AbstractTensor(4, 4, initializer=np.eye(4))
    w = AbstractTensor(9, 2, initializer=np.random.rand)
    x = AbstractTensor(4, 2, 3, symbol='x')
    y1 = AbstractTensor(4, 2, 3, symbol='y_1')

    with pytest.raises(RuntimeError):
        AbstractTensor(-1)
    with pytest.raises(RuntimeError):
        AbstractTensor('a', -2, -3)

    assert u.ndim == 1
    assert v.ndim == 2
    assert w.ndim == 2
    assert x.ndim == 3
    assert u.shape == (3,)
    assert v.shape == (4, 4)
    assert w.shape == (9, 2)
    assert x.shape == (4, 2, 3)

    for t in [u, v, w, x, y1]:
        assert isinstance(repr(t), str)
        assert isinstance(str(t), str)
        assert isinstance(t._repr_tex_(), str)
        assert isinstance(t._repr_html_(), str)
