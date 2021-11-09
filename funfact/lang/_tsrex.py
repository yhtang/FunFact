#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import asciitree
from funfact.util.iterable import as_namedtuple, as_tuple, flatten_if
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
        return TsrEx(
            P.index_notation(
                self.root,
                P.indices(
                    tuple([i.root for i in as_tuple(indices)])
                )
            )
        )

    def __rshift__(self, output_indices):
        return TsrEx(
            P.let(
                self.root,
                P.indices(
                    tuple([i.root for i in as_tuple(output_indices)])
                )
            )
        )


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
        symbol = spec[0]
        initializer = spec[1]
        size = initializer.shape
    elif len(spec) == 1 and _is_tensor(spec[0]):
        symbol = f'Anonymous_{AbstractTensor.n_nameless}'
        AbstractTensor.n_nameless += 1
        initializer = spec[0]
        size = initializer.shape
    elif isinstance(spec[0], str):
        symbol, *size = spec
    else:
        # internal format for anonymous symbols
        symbol = f'__{AbstractTensor.n_nameless}'
        AbstractTensor.n_nameless += 1
        size = spec

    return TsrEx(P.tensor(
        AbstractTensor(symbol, *size, initializer=initializer))
    )
