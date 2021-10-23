#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import typing
import asciitree
from funfact.util.iterable import as_namedtuple, flatten_if
from ._ast import _AST, _ASNode, Primitives as P
from ._interp_ascii import ASCIIInterpreter
from ._interp_latex import LatexInterpreter
from ._interp_init import InitializationInterpreter
from ._interp_base import MergeInterpreter
from ._interp_index_surv import IndexSurvivalInterpreter
from ._interp_eval import EvaluationInterpreter
from ._tensor import AbstractTensor, AbstractIndex


class TsrEx(_AST):

    _latex_intr = LatexInterpreter()
    _ascii_intr = ASCIIInterpreter()
    _init_intr = InitializationInterpreter()
    _merge_intr = MergeInterpreter()
    _idx_intr = IndexSurvivalInterpreter()
    _eval_intr = EvaluationInterpreter()
    _asciitree = asciitree.LeftAligned(
        traverse=as_namedtuple(
            'TsrExTraversal',
            get_root=lambda root: root,
            get_children=lambda node: list(
                filter(
                    lambda elem: isinstance(elem, _ASNode),
                    flatten_if(
                        node.__dict__.values(),
                        lambda elem: isinstance(elem, (list, tuple))
                    )
                )
            ),
            get_text=lambda node: node.payload
        ),
        draw=asciitree.drawing.BoxStyle(
            gfx={
                'UP_AND_RIGHT': u'\u2570',
                'HORIZONTAL': u'\u2500',
                'VERTICAL': u'\u2502',
                'VERTICAL_AND_RIGHT': u'\u251c'
            },
            horiz_len=2,
            label_space=0,
            label_format=' {}',
            indent=1
        )
    )

    def __or__(self, interpreter):
        return TsrEx(interpreter(self.root))

    @property
    def asciitree(self):
        return self._asciitree(self._ascii_intr(self.root))

    def evaluate(self):
        out_init = self | self._init_intr
        out_idx = self | self._idx_intr
        merged = self._merge_intr( out_init.root , out_idx.root )
        out = self._eval_intr( merged )
        return out[0]

    def _repr_html_(self):
        return f'''$${self._latex_intr(self.root)}$$'''

    def __add__(self, rhs):
        return self._as_tree(P.add(self.root, self._as_node(rhs)))

    def __radd__(self, lhs):
        return self._as_tree(P.add(self._as_node(lhs), self.root))

    def __sub__(self, rhs):
        return self._as_tree(P.sub(self.root, self._as_node(rhs)))

    def __rsub__(self, lhs):
        return self._as_tree(P.sub(self._as_node(lhs), self.root))

    def __mul__(self, rhs):
        return self._as_tree(P.mul(self.root, self._as_node(rhs)))

    def __rmul__(self, lhs):
        return self._as_tree(P.mul(self._as_node(lhs), self.root))

    def __div__(self, rhs):
        return self._as_tree(P.div(self.root, self._as_node(rhs)))

    def __rdiv__(self, lhs):
        return self._as_tree(P.div(self._as_node(lhs), self.root))

    def __neg__(self):
        return self._as_tree(P.neg(self.root))

    def __pow__(self, exponent):
        return self._as_tree(P.pow(self.root, self._as_node(exponent)))

    def __rpow__(self, base):
        return self._as_tree(P.pow(self._as_node(base)), self.root)

    def __getitem__(self, indices):
        assert isinstance(self.root, P.tensor)
        tsrnode = self.root
        if isinstance(indices, typing.Iterable):
            assert len(indices) == tsrnode.value.ndim,\
                f"Indices {indices} does not match the rank of tensor {self}."
            return TsrEx(
                P.index_notation(tsrnode, [i.root for i in indices])
            )
        else:
            assert 1 == tsrnode.value.ndim,\
                f"Indices {indices} does not match the rank of tensor {self}."
            return TsrEx(P.index_notation(tsrnode, (indices.root,)))


def index(symbol):
    return TsrEx(P.index(AbstractIndex(symbol)))


def indices(symbols):
    return [index(s) for s in re.split(r'[,\s]+', symbols)]


def tensor(*spec, initializer=None):
    '''Construct an abstract tensor using `spec`.

    Parameters
    ----------
    spec:
        Formats supported:

        * symbol, size...
        * size...

    initializer:
        Initialization distribution
    '''
    if isinstance(spec[0], str):
        symbol, *size = spec
    else:
        symbol = f'Anonymous_{AbstractTensor.n_nameless}'
        AbstractTensor.n_nameless += 1
        size = spec

    return TsrEx(P.tensor(
        AbstractTensor(symbol, *size, initializer=initializer))
    )