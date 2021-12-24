#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import copy
import dataclasses
import re
import sys
import asciitree
import functools
from funfact.backend import active_backend as ab
from funfact.util.iterable import as_namedtuple, as_tuple, flatten_if
from ._ast import _AST, _ASNode, Primitives as P
from .interpreter import (
    dfs_filter, ASCIIRenderer, LatexRenderer, IndexPropagator, ShapeAnalyzer,
    EinsteinSpecGenerator
)
from ._terminal import AbstractIndex, AbstractTensor


class ASCIITreeFactory:

    @staticmethod
    def _make_printer(*extra_fields):
        def _getattr_safe(obj, attr):
            try:
                return getattr(obj, attr)
            except AttributeError:
                return None
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
                    f'({v}: {_getattr_safe(node, v)})' for v in extra_fields
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
    _einspec_generator = EinsteinSpecGenerator()
    _index_propagator = IndexPropagator()
    _shape_analyzer = ShapeAnalyzer()

    @functools.lru_cache()
    def _repr_html_(self):
        return f'''$${self._latex_intr(self.root)}$$'''

    @property
    def asciitree(self):
        return self._asciitree_factory(self.root)

    @property
    @functools.lru_cache()
    def _static_analyzed(self):
        return self._einspec_generator(
            self._shape_analyzer(self._index_propagator(self.root))
        )

    @property
    def shape(self):
        return self._static_analyzed.shape

    @property
    def live_indices(self):
        return self._static_analyzed.live_indices

    @property
    def ndim(self):
        return len(self._static_analyzed.shape)

    @property
    def einspec(self):
        return self._static_analyzed.einspec


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
            self.root, _BaseEx(rhs).root, 6, 'sum', 'subtract', None
        ))

    def __rsub__(self, lhs):
        return EinopEx(P.ein(
            _BaseEx(lhs).root, self.root, 6, 'sum', 'subtract', None
        ))

    def __mul__(self, rhs):
        return EinopEx(P.ein(
            self.root, _BaseEx(rhs).root, 5, 'sum', 'multiply', None
        ))

    def __rmul__(self, lhs):
        return EinopEx(P.ein(
            _BaseEx(lhs).root, self.root, 5, 'sum', 'multiply', None
        ))

    def __truediv__(self, rhs):
        return EinopEx(P.ein(
            self.root, _BaseEx(rhs).root, 5, 'sum', 'divide', None
        ))

    def __rtruediv__(self, lhs):
        return EinopEx(P.ein(
            _BaseEx(lhs).root, self.root, 5, 'sum', 'divide', None
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

        indices = as_tuple(indices)
        live_old = tsrex.root.live_indices
        if len(indices) != len(live_old):
            raise SyntaxError(
                f'Incorrect number of indices. '
                f'Expects {len(live_old)}, '
                f'got {len(indices)}.'
            )

        for new_expr in indices:
            if new_expr.root.name != 'index':
                raise SyntaxError(
                    'Indices to a tensor expression must be abstract indices.'
                )
        live_new = [i.root.item for i in indices]

        index_map = dict(zip(live_old, live_new))
        # if a 'new' live index is already used as a dummy one, replace the
        # dummy usage with an anonymous index to avoid conflict.
        for n in dfs_filter(lambda n: n.name == 'index', tsrex.root):
            i = n.item
            if i not in live_old and i in live_new:
                index_map[i] = AbstractIndex()

        for n in dfs_filter(lambda n: n.name == 'index', tsrex.root):
            n.item = index_map.get(n.item, n.item)

        return tsrex | IndexPropagator()


class TranspositionMixin:
    '''transpose the axes by permuting the live indices into target indices.'''
    def __rshift__(self, indices):
        return TsrEx(P.tran(
            self.root,
            P.indices(tuple([i.root for i in as_tuple(indices)]))
        ))


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
                    tuple([i.root for i in as_tuple(indices or [])])
                )
            )
        )


class EinopEx(TsrEx):
    # override the `>>` behavior from TranspositionMixin
    def __rshift__(self, output_indices):
        tsrex = EinopEx(copy(self.root))
        tsrex.root.outidx = P.indices(
            tuple([i.root for i in as_tuple(output_indices)])
        )
        return tsrex


def index(symbol=None):
    '''Create an index variable.

    Args:
        symbol (str):
            A human-readable symbol.
            The symbol will be used for screen printing and LaTeX rendering.
            Must conform to the format described by the tsrex formal grammar.

    !!! note
        The symbol is only meant for human comprehension and is not used in
        equality/identity tests of the indices.

    Returns:
        TsrEx: A single-index tensor expression.
    '''
    return IndexEx(P.index(AbstractIndex(symbol), bound=False, kron=False))


def indices(spec):
    '''Create multiple index varaibles at once.

    Args:
        spec (int or str):

            - If `int`, indicate the number of 'anonymous' index variables to
            create. These anonymous variables will be numbered sequentially
            starting from 0, and are guaranteed to be unique within runtime.
            - If `str`, must be a
            comma/space-delimited list of symbols in the format as described in
            [funfact.index][].

    Returns:
        tuple: Multiple single-index tensor expressions.

    Example:

        >>> i, j = indices(2); i

        $$i$$

        >>> i, j, k = indices('i, j, k'); k

        $$k$$

        >>> I = indices(9); I[0]

        $$\\#_0$$
    '''
    if isinstance(spec, int) and spec >= 0:
        return [index() for i in range(spec)]
    elif isinstance(spec, str):
        return [index(s) for s in re.split(r'[,\s]+', spec)]
    else:
        raise RuntimeError(f'Cannot create indices from {spec}.')


def tensor(*spec, initializer=None, optimizable=None):
    '''Construct an abstract tensor using `spec`.

    Args:
        spec (multiple):
            Formats supported:

            * `symbol, size...`: a alphanumeric symbol followed by the size for
            each dimension.
            * `size...`: size of each dimension.
            * `symbol, tensor`: a alphanumeric symbol followed by a concrete
            tensor such as ``np.eye(3)`` or ``rand(10, 7)``.
            * `tensor`: a concrete tensor.

        initializer (callable):
            Initialization distribution

        optimizable (boolean):
            True/False flag indicating if a tensor leaf should be optimized.
            The default behavior is dependent on the input for `spec`:

            * if a size for each dimension is provided in `spec`, optimizable
            is True by default
            * if a concrete tensor is provided in `spec`, optimizable is False
            by default

            The default behavior can be overriden by user input.

    Returns:
        TsrEx: A tensor expression representing an abstract tensor object.
    '''
    if len(spec) == 2 and isinstance(spec[0], str) and ab.is_tensor(spec[1]):
        # name + concrete tensor
        symbol = spec[0]
        initializer = spec[1]
        size = initializer.shape
        if optimizable is None:
            optimizable = False
    elif len(spec) == 1 and ab.is_tensor(spec[0]):
        # concrete tensor only
        symbol = None
        initializer = spec[0]
        size = initializer.shape
        if optimizable is None:
            optimizable = False
    elif len(spec) >= 1 and isinstance(spec[0], str):
        # name + size
        symbol, *size = spec
        if optimizable is None:
            optimizable = True
    else:
        # size only
        symbol = None
        size = spec
        if optimizable is None:
            optimizable = True

    for d in size:
        if not (isinstance(d, int) and d > 0):
            raise RuntimeError(
                f'Tensor size must be positive integer, got {d} instead.'
            )

    return TensorEx(P.tensor(
        AbstractTensor(*size, symbol=symbol, initializer=initializer,
                       optimizable=optimizable))
                    )
