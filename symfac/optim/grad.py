#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn.functional
import torch.optim
from symfac.util.iterable import as_namedtuple


def gradient_descent(
    f, target, learning_rate, loss='mse_loss', algorithm='AdamW', tol=1e-6,
    max_nsteps=10000, return_history=False
):
    '''

    Parameters
    ----------
    f: callable
        A factorization object, which upon evaluation returns a reconstructed
        tensor, as created with :py:mod:`symfac.geneprog`.
    target: tensor
        The target tensor to be reconstructed.
    learning_rate: float
        Learning rate of the algorithm.
    loss: str or callable
        The loss function for comparing the difference between the target and
        the reconstructed tensor. Could be either a callable that accepts 2
        arguments, or a string that names a loss function in
        :py:mod:`torch.nn.functional`.
    algorithm: str or class
        The algorithm for carrying out the actual gradient descent. Could be
        either a PyTorch optimizer class or a string that names an optimizer in
        :py:mod:`torch.optim`.
    tol: float
        Terminate the process if the loss between consecutive steps is smaller
        than this value.
    max_nsteps: int
        Maximum number of steps to run.
    '''
    if isinstance(loss, str):
        try:
            loss = getattr(torch.nn.functional, loss)
        except AttributeError as e:
            raise AttributeError(
                f'The loss function \'{loss}\' does not exist in
                torch.nn.functional.'
            )
    try:
        loss(target, target)
    except Exception as e:
        raise AssertionError(
            f'The loss function cannot not accept two arguments:\n{e}'
        )

    if isinstance(algorithm, str):
        try:
            algorithm = getattr(torch.optim, algorithm)
        except AttributeError as e:
            raise AttributeError(
                f'The algorithm \'{algorithm}\' does not exist in torch.optim.'
            )
    try:
        opt = algorithm(parameters, lr=learning_rate)
    except Exception as e:
        raise AssertionError(
            'Invalid optimization algorithm:\n{e}'
        )

    best_K, best_loss = None, None
    if return_history:
        loss_history = []
    for step in range(max_nsteps):
        opt.zero_grad()
        output = f()
        total_loss = torch.tensor(0.0)
        batch_loss = []
        for j, K in enumerate(K_batch):
            loss = F.mse_loss(K, K0)
            total_loss += loss
            batch_loss.append(loss)
            if best_loss is None or loss < best_loss:
                best_params = factorization.parameters
                best_run = j
                best_loss = loss.detach().clone()
                best_K = K.detach().clone()
        loss_history.append(batch_loss)
        total_loss.backward()
        opt.step()
    
    return as_namedtuple(
        best_loss=best_loss,
        best_params=best_params,
        best_output=best_output,
    )

        loss_history = np.array(loss_history, dtype=np.float)
        print(f'{optname} final loss', loss_history[-1, :])
        print(f'{optname} best loss', best_loss)
        print(f'{optname} best params', best_params)
        print(f'best K\n{best_K.numpy()}')
        print(f'K0\n{K0.numpy()}')
        print('===========================================')
        plt.figure(figsize=(12, 8))
        for i, line in enumerate(np.log(loss_history).T):
            plt.plot(line, ls='dashed', lw=0.75, label=f'batch {i}')
        plt.plot(np.log(
            np.minimum.accumulate(
                np.minimum.reduce(loss_history, axis=1),
                axis=0
            )
        ), color='k', lw=1.25, ls='solid', label='BEST')
        plt.title(optname, fontsize=14)
        plt.ylabel('log-loss')
        plt.xlabel('training steps')
        plt.legend(loc='upper right', fontsize=12)
        plt.show()