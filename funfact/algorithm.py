import funfact.optim
import funfact.loss
from funfact import Factorization
from funfact.backend import active_backend as ab
import tqdm
import numpy as np


def factorize(tsrex, target, lr=0.1, tol=1e-6, max_steps=10000,
              optimizer='Adam', loss='mse_loss', nvec=1, stop_by='best',
              **kwargs):
    '''Gradient descent optimizer for functional factorizations.

    Parameters
    ----------
    tsrex: TsrEx
        A FunFact tensor expression.
    target
        Target data tensor
    lr
        Learning rate (default: 0.1)
    tol
        Convergence tolerance
    max_steps
        Maximum number of steps
    optimizer
        Name of optimizer
    loss
        Name of loss
    nvec
        Number of vectorizations
    stop_by
        'best', 'steps', int
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

    opt_fac = _Factorization(tsrex, nvec=nvec)
    best_fac = Factorization(tsrex, nvec=nvec)

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
