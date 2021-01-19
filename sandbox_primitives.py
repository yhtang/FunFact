#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
from abc import ABC
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from deap import gp
import ntf.primitives as P


def draw_tree(expr, ax=None):
    nodes, edges, labels = gp.graph(expr)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

    if ax is None:
        plt.figure(figsize=(9, 9))
        ax = plt.gca()
    nx.draw_networkx_nodes(g, pos, ax=ax)
    nx.draw_networkx_edges(g, pos, ax=ax)
    nx.draw_networkx_labels(g, pos, labels, ax=ax)


class MatExpr(ABC):
    pass


class ColVecExpr(ABC):
    pass


class RowVecExpr(ABC):
    pass


pset = P.FactorizationPrimitiveSet(MatExpr)
pset.add_primitive(
    name='matrix_add',
    action=lambda x, y: x + y,
    in_types=[MatExpr, MatExpr], ret_type=MatExpr
)
pset.add_primitive(
    name='matrix_exp_ewise',
    action=lambda m: torch.exp(m),
    in_types=[MatExpr], ret_type=MatExpr
)
pset.add_primitive(
    name='vector_outer_sub',
    action=lambda x, y: x[:, None] - y[None, :],
    in_types=[ColVecExpr, RowVecExpr], ret_type=MatExpr
)
pset.add_terminal(
    name='random_row_vector',
    action=lambda shape: torch.rand(shape[0], requires_grad=True),
    ret_type=RowVecExpr
)
pset.add_terminal(
    name='random_col_vector',
    action=lambda shape: torch.rand(shape[1], requires_grad=True),
    ret_type=ColVecExpr
)


np.random.seed(int(time.time()))

nrow = 4
ncol = 4

fig, axs = plt.subplots(nrow, ncol, figsize=(16, 16))
for i, row in enumerate(axs):
    for j, ax in enumerate(row):
        expr = pset.gen_expr(MatExpr, 5)
        draw_tree(expr, ax=ax)

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
