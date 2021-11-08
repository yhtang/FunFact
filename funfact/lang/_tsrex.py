#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import typing
import asciitree
from funfact.util.iterable import as_namedtuple, flatten_if
from funfact.util.typing import _is_tensor
from ._ast import _AST, _ASNode, Primitives as P
from .interpreter import ASCIIRenderer, LatexRenderer
from ._tensor import AbstractTensor, AbstractIndex


class TsrEx(_AST):

    _latex_intr = LatexRenderer()
    _ascii_intr = ASCIIRenderer()
    _asciitree = asciitree.LeftAligned(
        traverse=as_namedtuple(
            'TsrExTraversal',
            get_root=lambda root: root,
            get_children=lambda node: list(
                filter(
                    lambda elem: isinstance(elem, _ASNode),
                    flatten_if(
                        node.fields_fixed.values(),
                        lambda elem: isinstance(elem, (list, tuple))
                    )
                )
            ),
            get_text=lambda node: node.ascii
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

    @property
    def asciitree(self):
        return self._asciitree(self._ascii_intr(self.root))

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

        * symbol, size...: a alphanumeric symbol followed by the size for each
                           dimension.
        * size...: size of each dimension.
        * symbol, tensor: a alphanumeric symbol followed by a concrete tensor
                          such as ``np.eye(3)`` or ``rand(10, 7)``.
        * tensor: a concrete tensor.

    initializer:
        Initialization distribution

    Returns
    -------
    tsrex: TsrEx
        A tensor expression representing a single tensor object.
    '''
    if len(spec) == 2 and isinstance(spec[0], str) and _is_tensor(spec[1]):
        raise NotImplementedError()
    elif len(spec) == 1 and _is_tensor(spec[1]):
        raise NotImplementedError()
    elif isinstance(spec[0], str):
        symbol, *size = spec
    else:
        symbol = f'Anonymous_{AbstractTensor.n_nameless}'
        AbstractTensor.n_nameless += 1
        size = spec

    if hasattr(size[0], "__len__"):
        initializer = size[0]
        size = size[0].shape

    return TsrEx(P.tensor(
        AbstractTensor(symbol, *size, initializer=initializer))
    )
