#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
from unittest.mock import MagicMock as M
from funfact import active_backend as ab
from ._factorization import Factorization


def _test_factorization_props(fac):
    assert isinstance(fac, Factorization)
    assert hasattr(fac, '_tsrex')
    assert hasattr(fac, 'tsrex')
    assert hasattr(fac, 'shape')
    assert hasattr(fac, 'ndim')
    assert hasattr(fac, 'ndim')
    assert hasattr(fac, 'factors')
    assert hasattr(fac, 'all_factors')
    assert fac.forward() == fac()
    assert isinstance(fac.factors, Factorization._NodeView)
    assert isinstance(fac.all_factors, Factorization._NodeView)


def test_init_factory():
    fac = Factorization(M())
    _test_factorization_props(fac)
    fac = Factorization.from_tsrex(M(), dtype=M(), initialize=M())
    _test_factorization_props(fac)


def test_nodeview():
    nodes = [M(data=0), M(data=1)]
    nv = Factorization._NodeView('data', nodes)
    assert isinstance(nv.__repr__(), str)
    assert len(nv) == 2
    for i, node in enumerate(nv):
        assert i == node
    for i in range(len(nv)):
        assert nv[i] == i
        nv[i] = i + 10
        assert nv[i] == i + 10


@pytest.mark.parametrize('test_case', [
    (0, slice(0, 1, 2), slice(0, 1, 2)),
    (0, 1, slice(1, 2, None)),
    (0, -1, slice(-1, None, None)),
    (0, Ellipsis, None),
    (0, [0, 1], (0, 1)),
    (0, 1.2, RuntimeError)
])
def test_as_slice(test_case):
    i, axis,  result = test_case
    as_slice = Factorization._as_slice
    if isinstance(result, type) and issubclass(result, Exception):
        with pytest.raises(result):
            as_slice(i, axis)
    else:
        assert as_slice(i, axis) == result


def _gen_factors_mock(data, optimizable=True):
    root = M(data=data, decl=M(optimizable=optimizable))
    root.name = 'tensor'
    return Factorization(M(), _tsrex=M(root=root))


def test_factors():
    fac = _gen_factors_mock(1, True)
    assert len(fac.factors) == 1
    assert len(fac.all_factors) == 1
    assert fac.factors[0] == fac.all_factors[0]
    assert fac.factors[0] == 1
    fac.factors = [2]
    assert fac.factors[0] == 2
    fac = _gen_factors_mock(1, False)
    assert len(fac.factors) == 0
    assert len(fac.all_factors) == 1
    assert fac.all_factors[0] == 1


def test_penalties():
    def _prefer(data, *args):
        return ab.sum(data)

    root = M(data=ab.tensor([1, 2, 3]),
             decl=M(optimizable=True, prefer=_prefer))
    root.name = 'tensor'
    fac = Factorization(M(), _tsrex=M(root=root))
    assert fac.penalty() == 6


def test_get_set_item():
    root = M(data=1, decl=M(symbol='a', optimizable=True), ndim=1)
    root.name = 'tensor'
    fac = Factorization(M(), _tsrex=M(root=root))
    assert fac['a'] == 1
    fac['a'] = 2
    assert fac['a'] == 2
    with pytest.raises(AttributeError):
        fac['unknown-factor']
    with pytest.raises(AttributeError):
        fac['unknown-factor'] = 2
    with pytest.raises(IndexError):
        fac[0, 0]
    with pytest.raises(IndexError):
        fac[0, ...]
    #assert fac[...] == 1


# TODO: integration tests on top level
''' NEW TESTS W/O MAGICMOCK
import funfact as ff
from funfact import active_backend as ab


def test_instantiation():
    A = ff.tensor('A', 2, 2, initializer=ff.initializers.Ones)

    fac = ff.Factorization(A, test=1.0)
    assert fac.test == 1.0

    fac = ff.Factorization.from_tsrex(A, dtype=ab.float32)
    assert ab.allclose(fac.tsrex.root.data, ab.tensor([1.0]))
    assert fac.tsrex.root.data.dtype == ab.float32

    fac = ff.Factorization.from_tsrex(A, initialize=False)
    with pytest.raises(AttributeError):
        fac.tsrex.root.data
    assert isinstance(fac.tsrex, ff.lang._tsrex.TsrEx)
    assert fac.shape == A.shape
    assert fac.ndim == A.ndim


def test_factors_element_access():
    A = ff.tensor('A', 2, 2, optimizable=False)
    B = ff.tensor('B', 2, 2)
    tsrex = A @ B
    fac = ff.Factorization.from_tsrex(tsrex)
    assert len(fac.factors) == 1
    assert len(fac.all_factors) == 2
    fac.factors = ab.tensor([[1, 2], [3, 4]])
    ab.allclose(fac.factors[0], ab.tensor([[1, 2], [3, 4]]))
    fac.factors[0] = ab.tensor([[5, 6], [7, 8]])
    ab.allclose(fac.factors[0], ab.tensor([[5, 6], [7, 8]]))
    assert isinstance(fac.factors.__repr__(), str)
    assert ab.allclose(fac(), fac.forward())
    assert fac().shape == (2, 2)
    assert ab.allclose(fac['B'], fac.factors[0])
    with pytest.raises(AttributeError):
        fac['Unknown']
    fac['A'] = ab.tensor([[10, 20], [30, 40]])
    assert ab.allclose(fac['A'], ab.tensor([[10, 20], [30, 40]]))
    with pytest.raises(AttributeError):
        fac['Unknown'] = ab.tensor([-1])
    assert fac[0, 0] == fac()[0, 0]
    assert fac[-1, -1] == fac()[-1, -1]
    assert ab.allclose(ab.squeeze(fac[:, 0]), fac()[:, 0])
    assert ab.allclose(ab.squeeze(fac[1, ...]), fac()[1, ...])
    with pytest.raises(IndexError):
        fac[0, 0, 0]


def test_multiple_factors():
    A = ff.tensor('A', 2, 2)
    B = ff.tensor('B', 2, 2)
    tsrex = A @ B
    fac = ff.Factorization.from_tsrex(tsrex)
    i = 0
    for f in fac.factors:
        i += 1
    assert i == 2


def test_penalties():
    A = ff.tensor('A', 2, 2, prefer=ff.conditions.UpperTriangular())
    B = ff.tensor('B', 2, 2, prefer=ff.conditions.Unitary())
    tsrex = A @ B
    fac = ff.Factorization.from_tsrex(tsrex)
    assert fac.penalty() > 0.0
    assert fac.penalty(sum_leafs=False).shape == (2,)
    assert fac.penalty() == fac.penalty(sum_leafs=False)[0] + \
           fac.penalty(sum_leafs=False)[1]

'''

''' OLD TESTS
def test_elementwise():
    tol = 20 * np.finfo(np.float32).eps

    # matrix product
    A = tensor('A', 2, 2)
    B = tensor('B', 2, 2)
    i, j, k, m = indices('i, j, k, m')
    tsrex = A[i, j] * B[j, k]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (1, 0)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # one row
    idx = (1, slice(None))
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # one column
    idx = (slice(None), 0)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # outer product
    A = tensor('A', 10)
    B = tensor('B', 5)
    tsrex = A[i] * B[j]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (1, 0)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # slices
    idx = (slice(1, 6), slice(2, 4))
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # bound index in matrix product
    A = tensor('A', 2, 3)
    B = tensor('B', 3, 4)
    tsrex = A[i, j] * B[~j, k]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (1, 0, 1)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # slices
    idx = (slice(0, 2), slice(2, 3), 0)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # combination of different contractions
    A = tensor('A', 2, 3, 4)
    B = tensor('B', 4, 3, 2)
    tsrex = A[i, j, k] * B[k, ~j, m]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (0, 2, 1)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # slices
    idx = (1, slice(0, 2), 0)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f

    # Tucker decomposition
    T = tensor('T', 3, 3, 3)
    u1 = tensor('u_1', 4, 3)
    u2 = tensor('u_2', 5, 3)
    u3 = tensor('u_3', 6, 3)
    i1, i2, i3, k1, k2, k3 = indices('i_1, i_2, i_3, k_1, k_2, k_3')
    tsrex = T[k1, k2, k3] * u1[i1, k1] * u2[i2, k2] * u3[i3, k3]
    fac = Factorization.from_tsrex(tsrex)

    # one element
    idx = (0, 2, 1)
    full = fac()[idx]
    elementwise = fac[idx]
    assert pytest.approx(np.ravel(elementwise), tol) == np.ravel(full)

    # slices
    idx = (1, slice(0, 2), 0)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f
    idx = (slice(0, 3), slice(None), 2)
    full = fac()[idx]
    elementwise = fac[idx]
    for f, e in zip(np.ravel(full), np.ravel(elementwise)):
        assert pytest.approx(e, tol) == f


def test_Kronecker():
    tol = 2 * np.finfo(np.float32).eps
    dataA = np.reshape(np.arange(0, 6), (2, 3))
    dataB = np.reshape(np.arange(6, 15), (3, 3))
    A = tensor('A', dataA)
    B = tensor('B', dataB)
    i, j, k = indices('i, j, k')

    # regular Kronecker product
    tsrex = A[[*i, *j]] * B[i, j]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (6, 9)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f
    ref = np.kron(dataA, dataB)
    assert np.allclose(out, ref, tol)

    # Kronecker product along first axis (Khatri-Rao)
    tsrex = A[[*i, ~j]] * B[i, j]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (6, 3)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f
    ref = np.vstack([np.kron(dataA[:, k], dataB[:, k]) for k in
                    range(dataB.shape[1])]).T
    assert np.allclose(out, ref, tol)

    # Kronecker product along first axis, reduction second
    tsrex = A[[*i,  j]] * B[i, j]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (6,)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f

    # Matrix product
    tsrex = A[[i,   j]] * B[j, k]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (2, 3)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f
    ref = dataA @ dataB
    assert np.allclose(out, ref, tol)

    # No reduction
    tsrex = A[[i,  ~j]] * B[j, k]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (2, 3, 3)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f

    # Kronecker product inner axis
    tsrex = A[[i,  *j]] * B[j, k]
    fac = Factorization.from_tsrex(tsrex)
    out = fac()
    expected_shape = (2, 9, 3)
    for o, f, e in zip(out.shape, fac.shape, expected_shape):
        assert o == e
        assert o == f
'''
