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
    tsrex, target, lr=0.1, tol=1e-6, max_steps=10000, optimizer='Adam',
    loss='mse_loss', nvec=1, append=False, stop_by='first', returns='best',
    checkpoint_freq=50, dtype=None, penalty_weight=1.0, **kwargs
):
    '''Factorize a target tensor using the given tensor expression. The
    solution is found by minimizing the loss function between the original and
    approximate tensors using stochastic gradient descent.

    Args:
        tsrex (TsrEx): A tensor expression.
        target (tensor): The original tensor to approximate.
        lr (float): SGD learning rate.
        tol (float):  convergence tolerance.
        max_steps (int): maximum number of SGD steps to run.
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

            - If 'best', return the solution with the smallest loss.
            - If int `n`, return the top `n` instances.
            - If 'all', return all instances.

        checkpoint_freq (int >= 1): The frequency of convergence checking.

        dtype: The datatype of the factorization model (None, ab.dtype):

            - If None, the same data type as the target tensor is used.
            - If concrete dtype (float32, float64, complex64, complex128),
            that data type is used.

        penalty_weight (float) : weight of penalties relative to loss.

    Returns:
        *:
            - If `returns == 'best'`, return a factorization object of type
            [funfact.Factorization]() representing the best solution found.
            - If `returns == n`, return a list of factorization
            objects representing the best `n` solutions found.
            - If `returns == 'all'`, return a vectorized factorization object
            that represents all the solutions.
    '''

    if dtype is None:
        target = ab.tensor(target)
        dtype = target.dtype
    else:
        target = ab.tensor(target, dtype=dtype)

    @ab.autograd_decorator
    class _Factorization(Factorization, ab.AutoGradMixin):
        pass

    if isinstance(loss, str):
        try:
            loss = getattr(funfact.loss, loss)
        except AttributeError:
            raise AttributeError(
                f'The loss function \'{loss}\' does not exist in'
                'funfact.loss.'
            )
    try:
        loss(target, target, **kwargs)
    except Exception as e:
        raise AssertionError(
            f'The given loss function does not accept two arguments:\n{e}'
        )

    if isinstance(optimizer, str):
        try:
            optimizer = getattr(funfact.optim, optimizer)
        except AttributeError:
            raise AttributeError(
                f'The optimizer \'{optimizer}\' does not exist in'
                'funfact.optim.'
            )

    tsrex_vec = vectorize(tsrex, nvec, append=append)
    opt_fac = _Factorization.from_tsrex(tsrex_vec, dtype=dtype)

    try:
        opt = optimizer(opt_fac.factors, lr=lr, **kwargs)
    except Exception:
        raise AssertionError(
            'Invalid optimization algorithm:\n{e}'
        )

    def loss_and_penalty(model, target, sum_vec=True, **kwargs):
        loss_val = loss(model(), target, sum_vec=sum_vec, **kwargs)
        if penalty_weight > 0:
            return loss_val + penalty_weight * \
                   model.penalty(sum_leafs=True, sum_vec=sum_vec)
        else:
            return loss_val

    loss_and_grad = ab.loss_and_grad(loss_and_penalty, opt_fac, target,
                                     vectorized_along_last=append)

    # bookkeeping
    best_factors = [np.zeros_like(ab.to_numpy(x)) for x in opt_fac.factors]
    best_loss = np.ones(nvec) * np.inf
    converged = np.zeros(nvec, dtype=np.bool_)

    for step in tqdm.trange(max_steps):
        _, grad = loss_and_grad(opt_fac, target)
        with ab.no_grad():
            opt.step(grad)

            if step % checkpoint_freq == 0:
                # update best factorization
                curr_loss = ab.to_numpy(
                    loss_and_penalty(
                        opt_fac, target, sum_vec=False,
                        vectorized_along_last=append
                    )
                )
                better = np.flatnonzero(curr_loss < best_loss)
                best_loss = np.minimum(best_loss, curr_loss)
                for b, o in zip(best_factors, opt_fac.factors):
                    if append:
                        b[..., better] = o[..., better]
                    else:
                        b[better, ...] = o[better, ...]

                converged |= np.where(curr_loss < tol, True, False)
                if stop_by == 'first':
                    if np.any(converged):
                        break
                elif isinstance(stop_by, int):
                    if np.count_nonzero(converged) >= stop_by:
                        break
                else:
                    if stop_by is not None:
                        raise RuntimeError(
                            f'Invalid argument value for stop_by: {stop_by}'
                        )

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
        return best_fac
    else:
        raise RuntimeError(f'Invalid argument value for returns: {returns}')
