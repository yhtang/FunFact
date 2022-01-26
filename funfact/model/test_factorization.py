#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
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
        fac[0, 'A']
    with pytest.raises(IndexError):
        fac[0, 0, 0]
    with pytest.raises(IndexError):
        fac[5:6, 0]
    with pytest.raises(IndexError):
        fac[0:3, 0]


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
