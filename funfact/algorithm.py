#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tqdm
import funfact.optim
import funfact.loss
from funfact import Factorization
from funfact.backend import active_backend as ab
from funfact.vectorization import vectorize, view


def factorize(
    tsrex, target, optimizer='Adam', loss='MSE', lr=0.1, tol=1e-6,
    max_steps=10000, nvec=1, append=False, stop_by='first', returns='best',
    checkpoint_freq=50, dtype=None, penalty_weight=1.0
):
    '''Factorize a target tensor using the given tensor expression. The
    solution is found by minimizing the loss function between the original and
    approximate tensors using stochastic gradient descent.

    Args:
        tsrex (TsrEx): A tensor expression.
        target (tensor): The original tensor to approximate.
        optimizer (str or callable):

            - If `str`, must be one of the optimizers defined in
            [funfact.optim]().
            - If `callable`, can be any object that implements the interface of
            [funfact.optim.Optimizer]().

        loss (str or callable):

            - If `str`, must be one of the loss functions defined in
            [funfact.loss]().
            - If `callable`, can be any object that implements the interface of
            [funfact.loss.Loss]().

        lr (float): SGD learning rate.
        tol (float):  convergence tolerance.
        max_steps (int): maximum number of SGD steps to run.
        nvec (int): Number of parallel instances to compute.
        append (bool): If vectorizing axis is appended or prepended.
        stop_by ('first', int >= 1, or None):

            - If 'first', stop optimization as soon as one solution is
            found whose loss is less than `tol` when running multiple parallel
            instances.
            - If int `n`, stop optimization after n instances
            have found solutions with losses less than `tol`.
            - If None, always optimize for `max_steps` steps.

        returns ('best', int >= 1, or 'all'):

            - If 'best', returns the solution with the smallest loss.
            - If int `n` or 'all', returns a list of the top `n` or all of the
            instances sorted in ascending order by loss.

        checkpoint_freq (int >= 1): The frequency of convergence checking.

        dtype: The datatype of the factorization model (None, ab.dtype):

            - If None, the same data type as the target tensor is used.
            - If concrete dtype (float32, float64, complex64, complex128),
            that data type is used.

        penalty_weight (float) : Weight of penalties relative to loss.

    Returns:
        *:
            - If `returns == 'best'`, return a factorization object of type
            [funfact.Factorization]() representing the best solution found.
            - If `returns == n`, return a list of factorization
            objects representing the best `n` solutions found.
            - If `returns == 'all'`, return a vectorized factorization object
            that represents all the solutions.
    '''

    tsrex_vec = vectorize(tsrex, nvec, append=append)

    fac = ab.add_autograd(Factorization).from_tsrex(tsrex_vec, dtype=dtype)

    if dtype is None:
        target = ab.tensor(target)
        dtype = target.dtype
    else:
        target = ab.tensor(target, dtype=dtype)

    if isinstance(optimizer, str):
        try:
            optimizer = getattr(funfact.optim, optimizer)
        except AttributeError:
            raise RuntimeError(
                f'The optimizer \'{optimizer}\' does not exist in'
                'funfact.optim.'
            )
    try:
        opt = optimizer(fac.factors, lr=lr)
    except Exception:
        raise RuntimeError(
            'Invalid optimization algorithm:\n{e}'
        )

    if isinstance(loss, str):
        try:
            loss = getattr(funfact.loss, loss)
        except AttributeError:
            raise RuntimeError(
                f'The loss function \'{loss}\' does not exist in'
                'funfact.loss.'
            )
    if isinstance(loss, type):
        loss = loss()
    try:
        loss(target, target)
    except Exception as e:
        raise RuntimeError(
            f'A loss function must accept two arguments:\n{e}'
        )

    def loss_and_penalty(model, target, sum_vec=True):
        loss_val = loss(
            model(), target, sum_vec=sum_vec, vectorized_along_last=append
        )
        if penalty_weight > 0:
            return loss_val + penalty_weight * \
                   model.penalty(sum_leafs=True, sum_vec=sum_vec)
        else:
            return loss_val

    loss_and_grad = ab.loss_and_grad(loss_and_penalty, fac, target)

    if stop_by == 'first':
        stop_by = 1
    if not any((
        stop_by is None, isinstance(stop_by, int) and stop_by > 0
    )):
        raise RuntimeError(f'Invalid argument value for stop_by: {stop_by}')

    if not any((
        returns in ['best', 'all'], isinstance(returns, int) and returns > 0
    )):
        raise RuntimeError(f'Invalid argument value for returns: {returns}')

    # bookkeeping
    best_factors = [np.zeros_like(ab.to_numpy(x)) for x in fac.factors]
    best_loss = np.ones(nvec) * np.inf
    converged = np.zeros(nvec, dtype=np.bool_)

    for step in tqdm.trange(max_steps):
        _, grad = loss_and_grad(fac, target)
        with ab.no_grad():
            opt.step(grad)

            if step % checkpoint_freq == 0:
                # update best factorization
                curr_loss = ab.to_numpy(
                    loss_and_penalty(fac, target, sum_vec=False)
                )
                better = np.flatnonzero(curr_loss < best_loss)
                best_loss = np.minimum(best_loss, curr_loss)
                for b, o in zip(best_factors, fac.factors):
                    if append:
                        b[..., better] = o[..., better]
                    else:
                        b[better, ...] = o[better, ...]

                converged |= np.where(curr_loss < tol, True, False)
                if stop_by is not None:
                    if np.count_nonzero(converged) >= stop_by:
                        break

    best_fac = Factorization.from_tsrex(tsrex_vec, dtype=dtype)
    best_fac.factors = [ab.tensor(x) for x in best_factors]
    if returns == 'best':
        return view(best_fac, tsrex, np.argmin(best_loss), append)
    elif isinstance(returns, int):
        return [
            view(best_fac, tsrex, i, append) for i in
            np.argsort(best_loss)[:returns]
        ]
    elif returns == 'all':
        return [
            view(best_fac, tsrex, i, append) for i in np.argsort(best_loss)
        ]
