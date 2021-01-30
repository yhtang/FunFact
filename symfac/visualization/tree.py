#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import networkx as nx
from deap import gp


def draw_deap_expression(expr, ax=None):
    '''Plot the syntax tree of a prefix expression.

    Parameters
    ----------
    expr: iterable
        A prefix expression of grammatical primitives.
    ax: :py:class:`matplotlib.axes.Axes`
        The axes on which to draw the syntax tree. If None, a new figure and
        axes will be created.
    '''
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
