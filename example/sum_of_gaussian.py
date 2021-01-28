#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ntf.geneprog import PrimitiveSet
from ntf.visualization import draw_deap_expression


# define abstract types used in the factorization
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


Matrix = PrimitiveSet.new_type()
RowCol = PrimitiveSet.new_type()


# create the primitive set envelop
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


pset = PrimitiveSet(Matrix)


# create input primitives according to the target rank of the factorization
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


for i in range(3):
    @pset.add_terminal(name=f'rank{i}', ret_type=RowCol, params=['col', 'row'])
    def random_col_row_pair(p, n, m):
        # p.row = torch.normal(0.0, 1.0, [n], requires_grad=True)
        # p.col = torch.normal(0.0, 1.0, [m], requires_grad=True)
        p.row = torch.rand(n, requires_grad=True)
        p.col = torch.rand(m, requires_grad=True)
        return lambda: (p.row, p.col)


# create arithmetic primitives
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


@pset.add_primitive(ret_type=Matrix, in_types=[RowCol])
def gauss(p):
    return lambda uv: torch.exp(
        -0.5 * torch.square(uv[0][:, None] - uv[1][None, :])
    )


@pset.add_primitive(ret_type=Matrix, in_types=[Matrix], params=['a', 'b'])
def scale(p):
    p.a = torch.normal(0.0, 1.0, [1], requires_grad=True)
    p.b = torch.normal(0.0, 1.0, [1], requires_grad=True)
    return lambda M: M * p.a + p.b


@pset.add_primitive(ret_type=Matrix, in_types=[Matrix, Matrix])
def matrix_add(p):
    return lambda M1, M2: M1 + M2


# create an example expression direct from a string
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


expr = pset.from_string('''
matrix_add(
    gauss(rank0),
    gauss(rank1)
)''')


# visualize the expression
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# draw_deap_expression(expr)
# plt.show()


# now try to generate a toy problem:
# reconstruct K0 without knowing either f or any of U0, V0, U1, V1
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def f(u0, v0, u1, v1):
    # return (
    #     torch.exp(-0.5 * (u0[:, None] - v0[None, :])**2)
    #     + torch.exp(-0.5 * (u1[:, None] - v1[None, :])**2)
    # )
    return (
        torch.exp(-0.5 * torch.square(u0[:, None] - v0[None, :]))
        + torch.exp(-0.5 * torch.square(u1[:, None] - v1[None, :]))
    )


n = 3
m = 4
K0 = f(
    torch.rand(n), torch.rand(m),
    torch.rand(n), torch.rand(m),
)


# learn the factorization using gradienbt descent
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# create initial guesses
plt.figure(figsize=(15, 10))
lr = 0.1
for optname, optcls in [# ('SGD', optim.SGD),
                        ('Adam', optim.Adam),
                        # ('Adadelta', optim.Adadelta),
                        # ('Adagrad', optim.Adagrad),
                        # ('AdamW', optim.AdamW),
                        # ('ASGD', optim.ASGD),
                        # ('LBFGS', optim.LBFGS),
                        # ('RMSprop', optim.RMSprop),
                        ]:
    factorization = pset.instantiate(expr, n=n, m=m)
    opter = optcls(factorization.parameters, lr=lr)
    loss_history = []
    for _ in range(1000):
        opter.zero_grad()
        K = factorization()
        loss = F.mse_loss(K, K0)
        loss_history.append(loss)
        loss.backward()
        try:
            opter.step()
        except TypeError:
            opter.step(lambda: F.mse_loss(factorization(), K0))
    print(f'{optname} final loss', loss)
    print(f'K\n{factorization()}')
    print(f'K0\n{K0}')
    print('===========================================')
    plt.plot(np.log(loss_history), label=optname)
plt.ylabel('log-loss')
plt.xlabel('training steps')
plt.legend(loc='upper right', fontsize=12)
plt.show()
