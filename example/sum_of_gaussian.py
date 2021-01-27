#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools as it
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from deap import gp
from ntf.geneprog import PrimitiveSet
from ntf.visualization import draw_deap_expression


Matrix = PrimitiveSet.new_type()
ColRowPair = PrimitiveSet.new_type()

pset = PrimitiveSet(Matrix)

'''create inputs according to the target rank of the factorization'''
for i in range(3):
    @pset.add_terminal(
        name=f'rank_{i}', ret_type=ColRowPair,
        params=['col', 'row'], hyper_params=['shape']
    )
    def random_col_row_pair(self, shape):
        self.col = torch.normal(0.0, 1.0, shape[1:], requires_grad=True)
        self.row = torch.normal(0.0, 1.0, shape[:1], requires_grad=True)
        return lambda: (self.col, self.row)


@pset.add_primitive(ret_type=Matrix, in_types=[ColRowPair])
def gauss(self):
    return lambda uv: torch.exp(
        -0.5 * torch.square(uv[0][:, None] - uv[1][None, :])
    )


@pset.add_primitive(ret_type=Matrix, in_types=[Matrix], params=['a', 'b'])
def scale(self):
    self.a = torch.normal(0.0, 1.0, [1], requires_grad=True)
    self.b = torch.normal(0.0, 1.0, [1], requires_grad=True)
    return lambda M: M * self.a + self.b


@pset.add_primitive(ret_type=Matrix, in_types=[Matrix, Matrix])
def matrix_add(self):
    return lambda M1, M2: M1 + M2


expr = pset.from_string(
    'matrix_add(matrix_add(scale(gauss(rank_0)), gauss(rank_1)), gauss(rank_2))'
)
# draw_deap_expression(expr)
# plt.show()

factorization, _ = pset.instantiate(expr, shape=(10, 10))


# nrow = 2
# ncol = 2

# fig, axs = plt.subplots(nrow, ncol, figsize=(16, 16))
# for i, row in enumerate(axs):
#     for j, ax in enumerate(row):
#         expr = pset.gen_expr(5)
#         draw_tree(expr, ax=ax)


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
