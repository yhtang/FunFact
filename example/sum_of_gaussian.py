#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
from abc import ABC
import numpy as np
import torch
import matplotlib.pyplot as plt
from deap import gp
from ntf.primitives import FactorizationPrimitiveSet
from ntf.visualization import draw_deap_expression


class MatExpr(ABC):
    pass


class ColVecExpr(ABC):
    pass


class RowVecExpr(ABC):
    pass


class ColRowVecPair:
    def __init__(self):


class Node(ABC):

    @property
    @abstractmethod
    def parameters

    @property
    def parameters

class GassianOuterProduct(Node):
    def __init__(self, U, V):
        self.U = U
        self.V = V
        self.subtree = [self.U, self.V]

    def init():
        self.a = torch.rand(1, requires_grad=True)
        self.b = torch.rand(1, requires_grad=True)
        for x in self.subtree:
            x.randomize()

    def eval(self):
        u = U.eval()
        v = V.eval()
        return torch.exp(
            -0.5 * torch.square(
                u.forward()[:, None] - v[None, :])
        )

    @property
    def paramemters(self):
        return (self.a, self.b, self.U.parameters, self.V.parameters)


class ColVector:
    def __init__(self):
        

pset = FactorizationPrimitiveSet(
    ret_type=MatExpr,
    rank_types=[ColVecExpr, RowVecExpr],
    k=2
)

pset.add_primitive(
    name='matrix_add',
    action=lambda x, y: x + y,
    in_types=[MatExpr, MatExpr], ret_type=MatExpr
)
pset.add_primitive(
    name='gauss_outer',
    action=lambda u, v:
        torch.exp(-0.5 * torch.square(u[:, None] - v[None, :])),
    in_types=[ColVecExpr, RowVecExpr], ret_type=MatExpr
)
pset.add_ephemeral(

)


np.random.seed(int(time.time()))

expr = pset.gen_expr(5)
draw_deap_expression(expr)

# nrow = 2
# ncol = 2

# fig, axs = plt.subplots(nrow, ncol, figsize=(16, 16))
# for i, row in enumerate(axs):
#     for j, ax in enumerate(row):
#         expr = pset.gen_expr(5)
#         draw_tree(expr, ax=ax)

plt.show()

# while True:
#     try:
#         expr = pset.gen_expr()
#         break
#     except:
#         sys.stdout.write('.')
#         sys.stdout.flush()


# print(expr)
# print(type(expr))
# for symbol in expr:
#     print(type(symbol))



# factorization, _ = pset.instantiate(expr, (2, 2), random_state=0)

# print(factorization)

# M = factorization.forward()

# print(M)

# loss = torch.nn.MSELoss()

# L = loss(M, torch.zeros(M.shape))

# L.backward()

# # for optimizable in factorization:
# #     for dof in optimizable.state:
# #         dof -= alpha * dof.grad
