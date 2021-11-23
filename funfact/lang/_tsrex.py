#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dataclasses
import re
import sys
import asciitree
from funfact.util.iterable import as_namedtuple, as_tuple, flatten_if
from funfact.util.typing import _is_tensor
from ._ast import _AST, _ASNode, Primitives as P
from .interpreter import (
    dfs_filter, ASCIIRenderer, LatexRenderer, IndexPropagator
)
from ._terminal import AbstractIndex, AbstractTensor


class ASCIITreeFactory:

    @staticmethod
    def _make_printer(*extra_fields):
        return asciitree.LeftAligned(
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
                get_text=lambda node: node.ascii + ' ' + ' '.join([
                    f'({v}: {getattr(node, v)})' for v in extra_fields
                ])
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

    class ASCIITree:
        def __init__(self, root, factory):
            self._root = root
            self._factory = factory
            self._ascii_intr = ASCIIRenderer()

        def __repr__(self):
            return self._factory()(
                self._ascii_intr(self._root)
            )

        def __call__(self, *fields, stdout=True):
            ascii = self._factory(*fields)(
                self._ascii_intr(self._root)
            )
            if stdout:
                sys.stdout.write(ascii)
                sys.stdout.flush()
            else:
                return ascii

    def __call__(self, root):
        return self.ASCIITree(root, self._make_printer)


class _BaseEx(_AST):

    _latex_intr = LatexRenderer()
    _asciitree_factory = ASCIITreeFactory()

    def _repr_html_(self):
        return f'''$${self._latex_intr(self.root)}$$'''

    @property
    def asciitree(self):
        return self._asciitree_factory(self.root)


class ArithmeticMixin:

    def __add__(self, rhs):
        return EinopEx(P.ein(
            self.root, _BaseEx(rhs).root, 6, 'sum', 'add', None
        ))

    def __radd__(self, lhs):
        return EinopEx(P.ein(
            _BaseEx(lhs).root, self.root, 6, 'sum', 'add', None
        ))

    def __sub__(self, rhs):
        return EinopEx(P.ein(
            self.root, _BaseEx(rhs).root, 6, 'sum', 'sub', None
        ))

    def __rsub__(self, lhs):
        return EinopEx(P.ein(
            _BaseEx(lhs).root, self.root, 6, 'sum', 'sub', None
        ))

    def __mul__(self, rhs):
        return EinopEx(P.ein(
            self.root, _BaseEx(rhs).root, 5, 'sum', 'mul', None
        ))

    def __rmul__(self, lhs):
        return EinopEx(P.ein(
            _BaseEx(lhs).root, self.root, 5, 'sum', 'mul', None
        ))

    def __div__(self, rhs):
        return EinopEx(P.ein(
            self.root, _BaseEx(rhs).root, 5, 'sum', 'div', None
        ))

    def __rdiv__(self, lhs):
        return EinopEx(P.ein(
            _BaseEx(lhs).root, self.root, 5, 'sum', 'div', None
        ))

    def __neg__(self):
        return TsrEx(P.neg(self.root))

    def __pow__(self, exponent):
        return TsrEx(P.pow(self.root, _BaseEx(exponent).root))

    def __rpow__(self, base):
        return TsrEx(P.pow(_BaseEx(base).root, self.root))


class IndexRenamingMixin:
    '''Rename the free indices of a tensor expression.'''

    def __getitem__(self, indices):

        tsrex = self | IndexPropagator()

        if len(indices) != len(tsrex.root.live_indices):
            raise SyntaxError(
                f'Incorrect number of indices. '
                f'Expects {len(tsrex.root.live_indices)}, '
                f'got {len(indices)}.'
            )

        index_map = {}
        for old, new_expr in zip(tsrex.root.live_indices, indices):
            if new_expr.root.name != 'index':
                raise SyntaxError(
                    'Indices to a tensor expression must be abstract indices.'
                )
            index_map[old] = new_expr.root.item

        for n in dfs_filter(lambda n: n.name == 'index', tsrex.root):
            n.item = index_map.get(n.item, n.item)

        return tsrex | IndexPropagator()


class TranspositionMixin:
    '''transpose the axes by permuting the live indices into target indices.'''
    @property
    def T(self):
        return self._T(self.root)

    class _T(_BaseEx):
        def __getitem__(self, indices):
            return TsrEx(
                P.tran(self.root,
                       P.indices(tuple([i.root for i in as_tuple(indices)])))
            )


class TsrEx(_BaseEx, ArithmeticMixin, IndexRenamingMixin, TranspositionMixin):
    '''A general tensor expression'''
    pass


class IndexEx(_BaseEx):
    def __invert__(self):
        '''Implements the `~i` syntax.'''
        return IndexEx(dataclasses.replace(self.root, bound=True, kron=False))

    def __iter__(self):
        '''Implements the `*i` syntax.'''
        yield IndexEx(dataclasses.replace(self.root, bound=False, kron=True))


class TensorEx(_BaseEx):
    def __getitem__(self, indices):
        return TsrEx(
            P.index_notation(
                self.root,
                P.indices(
                    tuple([i.root for i in as_tuple(indices)])
                )
            )
        )


class EinopEx(TsrEx):
    def __rshift__(self, output_indices):
        self.root.outidx = P.indices(
            tuple([i.root for i in as_tuple(output_indices)])
        )
        return self


def index(symbol=None):
    return IndexEx(P.index(AbstractIndex(symbol), bound=False, kron=False))


def indices(spec):
    if isinstance(spec, int):
        return [index() for i in range(spec)]
    elif isinstance(spec, str):
        return [index(s) for s in re.split(r'[,\s]+', spec)]
    else:
        raise RuntimeError(f'Cannot create indices from {spec}.')


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
    tsrex: _BaseEx
        A tensor expression representing a single tensor object.
    '''
    if len(spec) == 2 and isinstance(spec[0], str) and _is_tensor(spec[1]):
        # name + concrete tensor
        symbol = spec[0]
        initializer = spec[1]
        size = initializer.shape
    elif len(spec) == 1 and _is_tensor(spec[0]):
        # concrete tensor only
        symbol = None
        initializer = spec[0]
        size = initializer.shape
    elif isinstance(spec[0], str):
        # name + size
        symbol, *size = spec
    else:
        # size only
        symbol = None
        size = spec

    return TensorEx(P.tensor(
        AbstractTensor(*size, symbol=symbol, initializer=initializer))
    )
