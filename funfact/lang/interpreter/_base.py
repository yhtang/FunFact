#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Tuple
from funfact.lang._ast import _ASNode, _AST, Primitives as P
from funfact.lang._terminal import AbstractIndex, AbstractTensor, LiteralValue
from funfact.util.iterable import flatten_if


'''An interpreter traverses an abstract syntax tree (AST) in a depth-first
manner and perform corresponding actions for each of the node. Applying
different interpreters to the same AST lead to different results, all of
which conform to yet reflects the different facets of the AST.

**See definitions of the primitives in the `Primitives` class.**'''


def _deep_apply(f, value, *args, **kwargs):
    if isinstance(value, (list, tuple)):
        return [_deep_apply(f, elem, *args, **kwargs) for elem in value]
    elif isinstance(value, _ASNode):
        return f(value, *args, **kwargs)
    else:
        return value


def _deep_apply_batch(f, *values):
    head = values[0]
    if isinstance(head, (list, tuple)):
        return [_deep_apply_batch(f, *elem) for elem in zip(*values)]
    elif isinstance(head, _ASNode):
        return f(*values)
    else:
        return head


def _as_payload(*k):
    if len(k) == 1:
        def wrapper(f):
            def wrapped_f(*args, **kwargs):
                return k[0], f(*args, **kwargs)
            return wrapped_f
        return wrapper
    else:
        def wrapper(f):
            def wrapped_f(*args, **kwargs):
                return dict(zip(k, f(*args, **kwargs)))
            return wrapped_f
        return wrapper


def _emplace(node, payload):
    if isinstance(payload, dict):
        node.__dict__.update(**payload)
    elif isinstance(payload, list):
        for key, value in payload:
            setattr(node, key, value)
    elif isinstance(payload, tuple) and len(payload) == 2:
        setattr(node, *payload)
    elif payload is None:
        pass
    else:
        raise TypeError(f'Unrecognizable type for payload {payload}')


class ROOFInterpreter(ABC):
    '''A ROOF (Read-Only On-the-Fly) interpreter traverses an AST for one pass
    and produces the final outcome without altering the AST. Intermediates are
    passed as return values between the traversing levels. Its primitive rules
    may still accept a 'payload' argument, which could be potentially produced
    by another transcribe interpreter.'''

    @abstractmethod
    def abstract_index_notation(
        self, tensor: Any, indices: Iterable[Any], **payload
    ):
        pass

    @abstractmethod
    def abstract_binary(
        self, lhs: Any, rhs: Any, precedence: int, operator: str, **payload
    ):
        pass

    @abstractmethod
    def literal(self, value: LiteralValue, **payload):
        pass

    @abstractmethod
    def tensor(self, decl: AbstractTensor, **payload):
        pass

    @abstractmethod
    def index(self, item: AbstractIndex, bound: bool, kron: bool, **payload):
        pass

    @abstractmethod
    def indices(self, items: AbstractIndex, **payload):
        pass

    @abstractmethod
    def indexed_tensor(
        self, tensor: Any, indices: Iterable[Any], **payload
    ):
        pass

    @abstractmethod
    def call(self, f: str, x: Any, **payload):
        pass

    @abstractmethod
    def neg(self, x: Any, **payload):
        pass

    @abstractmethod
    def elem(self, lhs: Any, rhs: Any, precedence: int, operator: str,
             **payload):
        pass

    @abstractmethod
    def ein(self, lhs: Any, rhs: Any, precedence: int, reduction: str,
            pairwise: str, outidx: Any, **payload):
        pass

    @abstractmethod
    def tran(self, src: Any, indices: Iterable[Any]):
        pass

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        fields_fixed = {
            name: _deep_apply(self, value, node)
            for name, value in node.fields_fixed.items()
        }
        rule = getattr(self, node.name)
        return rule(**fields_fixed, **node.fields_payload)

    def __ror__(self, tsrex: _AST):
        return self(tsrex.root)


class TranscribeInterpreter(ABC):
    '''A transcribe interpreter creates a modified copy of an AST while
    traversing it.'''

    class TraversalOrder(Enum):
        '''A post-order transcriber acts on the children of a node before
        acting on the node itself.'''
        PRE = 0
        POST = 1

    _traversal_order: TraversalOrder

    @abstractmethod
    def abstract_index_notation(
        self, tensor: P.Numeric, indices: P.indices, **payload
    ):
        pass

    @abstractmethod
    def abstract_binary(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **payload
    ):
        pass

    @abstractmethod
    def literal(self, value: LiteralValue, **payload):
        pass

    @abstractmethod
    def tensor(self, decl: AbstractTensor, **payload):
        pass

    @abstractmethod
    def index(self, item: AbstractIndex, bound: bool, kron: bool, **payload):
        pass

    @abstractmethod
    def indices(self, items: Tuple[AbstractIndex], **payload):
        pass

    @abstractmethod
    def indexed_tensor(
        self, tensor: P.Numeric, indices: P.indices, **payload
    ):
        pass

    @abstractmethod
    def call(self, f: str, x: P.Tensorial, **payload):
        pass

    @abstractmethod
    def neg(self, x: P.Numeric, **payload):
        pass

    @abstractmethod
    def elem(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, operator: str,
        **payload
    ):
        pass

    @abstractmethod
    def ein(
        self, lhs: P.Numeric, rhs: P.Numeric, precedence: int, reduction: str,
        pairwise: str, outidx: Optional[P.indices], **payload
    ):
        pass

    @abstractmethod
    def tran(self, src: P.Numeric, indices: P.indices):
        pass

    def _eval(self, node):
        return getattr(self, node.name)(**node.fields)

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        node = copy.copy(node)

        def _do_node():
            payload = self._eval(node)
            _emplace(node, payload)

        def _do_children():
            for name, value in node.fields_fixed.items():
                setattr(node, name, _deep_apply(self, value, node))

        if self._traversal_order == self.TraversalOrder.PRE:
            _do_node()
            _do_children()
        elif self._traversal_order == self.TraversalOrder.POST:
            _do_children()
            _do_node()
        return node

    def __ror__(self, tsrex: _AST):
        return type(tsrex)(self(tsrex.root))


class RewritingTranscriber(TranscribeInterpreter):
    '''A rewriting transcriber may either rewrite the payloads of a node, or
    replace the entire node alltogether.'''

    def __call__(self, node, parent=None, depth=float('inf')):

        node = copy.copy(node)

        if depth > 0:  # do children
            for name, value in node.fields_fixed.items():
                setattr(
                    node, name, _deep_apply(self, value, node, depth - 1)
                )

        while True:  # iterative rewriting
            value = self._eval(node)
            if not isinstance(value, _ASNode):
                break
            node = value

        _emplace(node, value)

        return node

    def __ror__(self, tsrex: _AST):
        return type(tsrex)(self(tsrex.root))


class PayloadMerger:
    '''The payload merger combines several homologus ASTs by concatenating the
    payloads of each group of same-place nodes as a tuple.'''
    def __call__(self, *nodes: _ASNode):
        head = copy.copy(nodes[0])
        for name in head.fields_fixed:
            setattr(head, name, _deep_apply_batch(
                self,
                *[getattr(n, name) for n in nodes]
            ))
        for n in nodes:
            head.fields.update(n.fields_payload)
        return head

    def __ror__(self, tsrex_list: Iterable[_AST]):
        return type(tsrex_list[0])(self(*[tsrex.root for tsrex in tsrex_list]))


def dfs(node: _ASNode):
    '''Returns an iterator that loop over all nodes in an AST in a depth-first
    manner.'''

    for child in flatten_if(
        node.fields_fixed.values(),
        lambda elem: isinstance(elem, (list, tuple))
    ):
        if isinstance(child, _ASNode):
            yield from dfs(child)
        yield node


def dfs_filter(function: Callable[[_ASNode], bool], node: _ASNode):
    '''Returns an iterator that loop over all nodes in an AST in a depth-first
    manner for which `function` evaluates to true.'''

    for child in flatten_if(
        node.fields_fixed.values(),
        lambda elem: isinstance(elem, (list, tuple))
    ):
        if isinstance(child, _ASNode):
            yield from dfs_filter(function, child)

    if function(node) is True:
        yield node
