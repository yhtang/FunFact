import funfact.optim
import funfact.loss
from funfact import Factorization
from funfact.backend import active_backend as ab
import tqdm
import numpy as np


def factorize(
    tsrex, target, lr=0.1, tol=1e-6, max_steps=10000, optimizer='Adam',
    loss='mse_loss', nvec=1, stop_by='best', **kwargs
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
        stop_by ('best', 'steps', or int):

            - If 'best', the function will return the first solution whose loss
            is less than `tol` when running multiple parallel instances.
            - If it is an integer `n`, the function will return after finding
            `n` solutions with losses less than `tol`.
            - If `steps`, only returns after completing `max_steps` steps.

    Returns:
        *:
            - If `stop_by == 'best'`, return a factorization object of type
            [funfact.Factorization]() representing the best solution found.
            - If `stop_by` is an integer `n`, return a list of factorization
            objects representing the top `n` solutions found.
            - If `stop_by == 'steps'`, return a vectorized factorization object
            that represents all the solutions.

    '''

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

    opt_fac = _Factorization.from_tsrex(tsrex, nvec=nvec)
    best_fac = Factorization.from_tsrex(tsrex, nvec=nvec)

    try:
        opt = optimizer(opt_fac.factors, lr=lr, **kwargs)
    except Exception:
        raise AssertionError(
            'Invalid optimization algorithm:\n{e}'
        )

    @ab.jit
    def _loss(fac, target):
        return loss(fac(), target)
    gradient = ab.jit(ab.grad(_loss))

    best_loss = np.ones(nvec) * np.inf
    converged = np.zeros(nvec)
    pbar = tqdm.tqdm(total=max_steps + 1)

    for step in range(max_steps):
        pbar.update(1)
        opt.step(gradient(opt_fac, target).factors)

        if step % round(max_steps/20) == 0:
            # update best factorization
            curr_loss = loss(opt_fac(), target, sum_vec=False)
            new_best = []
            for b, o in zip(best_fac.factors, opt_fac.factors):
                for i, l in enumerate(zip(curr_loss, best_loss)):
                    if l[0] < l[1]:
                        if b.shape[-1] == 1:
                            # TODO: this weird way of updating is required
                            # by JAX
                            b = b.at[..., 0].set(o[..., 0])
                        else:
                            b = b.at[..., i].set(o[..., i])
                        if l[0] < tol:
                            converged[i] = 1
                new_best.append(b)
            best_fac.factors = new_best

            if stop_by == 'best':
                if np.sum(converged) > 1:
                    pbar.update(max_steps - step)
                    break
            elif isinstance(stop_by, int):
                if np.sum(converged) > stop_by:
                    pbar.update(max_steps - step)
                    break
    pbar.close()

    if stop_by == 'best':
        return best_fac.view(np.argmin(best_loss))
    elif stop_by == 'steps':
        return best_fac
    elif isinstance(stop_by, int):
        sort_idx = np.argsort(best_loss)
        return [best_fac.view(sort_idx[i]) for i in range(stop_by)]
    else:
        raise ValueError(f'Unsupported value for stop_by: {stop_by}')
