from jax import grad, jit
import funfact.optim
import funfact.loss
from funfact import Factorization
import tqdm


def factorize(tsrex, target, lr=0.1, tol=1e-4, max_steps=10000,
              optimizer='Adam', loss='mse_loss', nvec=1, **kwargs):
    '''Gradient descent optimizer for functional factorizations.

    Parameters
    ----------
    tsrex: TsrEx
        A FunFact tensor expression.
    target
        Target data tensor
    lr
        Learning rate (default: 0.1)
    beta1
        First order moment (default: 0.9)
    beta2
        Second order moment (default: 0.999)
    epsilon
        default: 1e-7
    nsteps
        Number of steps in gradient descent (default:10000)
    '''

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
    fac = Factorization(tsrex, nvec=nvec)

    try:
        opt = optimizer(fac.factors, lr=lr, **kwargs)
    except Exception:
        raise AssertionError(
            'Invalid optimization algorithm:\n{e}'
        )

    @jit
    def _loss(fac, target):
        return loss(fac(), target)
    gradient = jit(grad(_loss))

    def progressbar(n):
        return tqdm.trange(
                n, miniters=None, mininterval=0.25, leave=True
            )

    for step in progressbar(max_steps):
        opt.step(gradient(fac, target).factors)

    return fac
