#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dataclasses
import re
import sys
import asciitree
import functools
from numbers import Number
from plum import Dispatcher
from funfact.backend import active_backend as ab
from funfact.util.iterable import as_namedtuple, as_tuple, flatten_if
from ._ast import _AST, _ASNode, Primitives as P
from .interpreter import (
    ASCIIRenderer,
    Compiler,
    EinsteinSpecGenerator,
    IndexnessAnalyzer,
    LatexRenderer,
)
from ._terminal import LiteralValue, AbstractIndex, AbstractTensor
from funfact.conditions import NoCondition


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
                get_text=lambda node: '{}: {} '.format(
                    node.name, node.ascii
                ) + ' '.join([
                    '({f}: {s})'.format(
                        f=f,
                        s=re.sub('\n', '', str(_getattr_safe(node, f)))
                    ) for f in extra_fields
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
    _compiler = Compiler()
    _einspec_generator = EinsteinSpecGenerator()
    _indexness_analyzer = IndexnessAnalyzer()

    @functools.lru_cache()
    def _repr_html_(self):
        '''LaTeX rendering of the tensor expression for Jupyter notebook.'''
        return f'''$${self._latex_intr(self.root)}$$'''

    @property
    def asciitree(self):
        '''A ASCII-based tree visualization of the tensor expression.
        This property can also be treated as a callable that accepts a list of
        string-valued attribute names to be displayed for each node.

        Examples:
            >>> import funfact as ff
            >>> a = ff.tensor('a', 2, 3)
            >>> b = ff.tensor('b', 3, 5)
            >>> i, j, k = ff.indices('i, j, k')
            >>> tsrex = a[i, j] * b[j, k]
            >>> tsrex.asciitree
            sum:multiply
            ├── a[i,j]
            │   ├──
            │   ╰── i,j
            │       ├── i
            │       ╰── j
            ╰── b[j,k]
                ├── b
                ╰── j,k
                    ├── j
                    ╰── k

            >>> fac = ff.Factorization.from_tsrex(tsrex)
            >>> fac.tsrex.asciitree('data')
            sum:multiply (data: None)
            ├── a[i,j] (data: None)
            │   ├── a (data: [[ 0.573 -0.406  0.164]
            │   │             [-0.492  0.128  1.428]])
            │   ╰── i,j (data: None)
            │       ├── i (data: None)
            │       ╰── j (data: None)
            ╰── b[j,k] (data: None)
                ├── b (data: [[-0.527  0.933  0.218 -1.49  -0.976]
                │             [ 0.028  0.285 -0.701 -1.753 -2.313]
                │             [-0.371 -0.406  0.374  1.17   1.372]])
                ╰── j,k (data: None)
                    ├── j (data: None)
                    ╰── k (data: None)
        '''
        return self._asciitree_factory(self.root)

    @property
    @functools.lru_cache()
    def _static_analyzed(self):
        return (self |
                self._indexness_analyzer |
                self._compiler |
                self._einspec_generator).root

    @property
    def shape(self):
        '''Shape of the evaluation result'''
        return self._static_analyzed.shape

    @property
    def ndim(self):
        '''Dimensionality of the evaluation result'''
        return len(self._static_analyzed.shape)

    @property
    def live_indices(self):
        '''Surviving indices that are not eliminated during tensor
        contractions.'''
        return self._static_analyzed.live_indices

    @property
    def einspec(self):
        '''NumPy-style einsum specification of the top level operation.'''
        return self._static_analyzed.einspec


def _as_node(x):
    if isinstance(x, _ASNode):
        return x
    elif isinstance(x, _AST):
        return x.root
    elif isinstance(x, Number):
        return P.literal(value=LiteralValue(x))
    else:
        raise TypeError(
            f'Cannot use {x} of type {type(x)} in a tensor expression.'
        )


def as_tsrex(f):
    def wrapper(*args, **kwargs):
        return TsrEx(f(*args, **kwargs))
    return wrapper


def yield_tsrex(f):
    def wrapper(*args, **kwargs):
        for n in f(*args, **kwargs):
            yield TsrEx(n)
    return wrapper


class SyntaxOverloadMixin:

    @as_tsrex
    def __neg__(self):
        return _neg(_as_node(self))

    @as_tsrex
    def __add__(self, rhs):
        return _binary(_as_node(self), _as_node(rhs), 6, 'add')

    @as_tsrex
    def __sub__(self, rhs):
        return _binary(_as_node(self), _as_node(rhs), 6, 'subtract')

    @as_tsrex
    def __mul__(self, rhs):
        return _binary(_as_node(self), _as_node(rhs), 5, 'multiply')

    @as_tsrex
    def __matmul__(self, rhs):
        return _binary(_as_node(self), _as_node(rhs), 5, 'matmul')

    @as_tsrex
    def __truediv__(self, rhs):
        return _binary(_as_node(self), _as_node(rhs), 5, 'divide')

    @as_tsrex
    def __pow__(self, rhs):
        return _binary(_as_node(self), _as_node(rhs), 3, 'float_power')

    @as_tsrex
    def __and__(self, rhs):
        return _binary(_as_node(self), _as_node(rhs), 5, 'kron')

    @as_tsrex
    def __radd__(self, lhs):
        return _binary(_as_node(lhs), _as_node(self), 6, 'add')

    @as_tsrex
    def __rsub__(self, lhs):
        return _binary(_as_node(lhs), _as_node(self), 6, 'subtract')

    @as_tsrex
    def __rmul__(self, lhs):
        return _binary(_as_node(lhs), _as_node(self), 5, 'multiply')

    @as_tsrex
    def __rmatmul__(self, lhs):
        return _binary(_as_node(lhs), _as_node(self), 5, 'matmul')

    @as_tsrex
    def __rtruediv__(self, lhs):
        return _binary(_as_node(lhs), _as_node(self), 5, 'divide')

    @as_tsrex
    def __rpow__(self, lhs):
        return _binary(_as_node(lhs), _as_node(self), 3, 'float_power')

    @as_tsrex
    def __rand__(self, lhs):
        return _binary(_as_node(lhs), _as_node(self), 5, 'kron')

    @as_tsrex
    def __getitem__(self, indices):
        return _getitem(_as_node(self), indices)

    @as_tsrex
    def __rshift__(self, indices):
        return _rshift(_as_node(self), indices)

    @as_tsrex
    def __invert__(self):
        return _invert(_as_node(self))

    @yield_tsrex
    def __iter__(self):
        return _iter(_as_node(self))


class TsrEx(
    _BaseEx,
    SyntaxOverloadMixin,
):
    '''A expression of potentially nested and composite tensor computations.

    Supported operations between tensor expressions include:

    - Elementwise: addition, subtraction, multiplication, division,
    exponentiation
    - Einstein summation and generalized semiring operations
    - Kronecker product
    - Transposition
    - Function evaluation

    For more details, see the corresponding
    [user guide page](../../pages/user-guide/tsrex).
    '''


_dispatch = Dispatcher()


@_dispatch
def _binary(lhs: _ASNode, rhs: _ASNode, precedence, operator):
    return P.abstract_binary(lhs, rhs, precedence, operator)


@_dispatch
def _neg(node: _ASNode):
    return P.neg(node)


@_dispatch
def _invert(node: P.index):
    '''Implements the `~i` syntax.'''
    return dataclasses.replace(node, bound=True, kron=False)


@_dispatch
def _iter(node: P.index):
    '''Implements the `*i` syntax.'''
    yield dataclasses.replace(node, bound=False, kron=True)


@_dispatch
def _getitem(node: _ASNode, indices):  # noqa: F811
    '''create index notation'''
    return P.abstract_index_notation(
        node,
        P.indices(tuple([i.root for i in as_tuple(indices or [])]))
    )


@_dispatch
def _rshift(node: _ASNode, indices):  # noqa: F811
    '''transpose or einsum output specification'''
    return P.abstract_dest(
        node,
        P.indices(tuple([i.root for i in as_tuple(indices)]))
    )


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
    return TsrEx(P.index(AbstractIndex(symbol), bound=False, kron=False))


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


def tensor(*spec, initializer=None, optimizable=None, prefer=None):
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

        prefer (callable):
            Condition evaluated on tensor as a penalty term. Only considered if
            optimizable is set to True or defaults to True, otherwise it is
            ignored.

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
    if optimizable and prefer is None:
        prefer = NoCondition()

    return TsrEx(
        P.tensor(
            AbstractTensor(
                *size, symbol=symbol, initializer=initializer,
                optimizable=optimizable, prefer=prefer
            )
        )
    )
