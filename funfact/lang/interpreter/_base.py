#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy
from numbers import Real
from typing import Any, Callable, Iterable, Optional, Tuple, Union
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


class ROOFInterpreter(ABC):
    '''A ROOF (Read-Only On-the-Fly) interpreter traverses an AST for one pass
    and produces the final outcome without altering the AST. Intermediates are
    passed as return values between the traversing levels. Its primitive rules
    may still accept a 'payload' argument, which could be potentially produced
    by another transcribe interpreter.'''

    @abstractmethod
    def noop(self):
        '''wildcard method for no-operations.'''

    def literal(self, value: LiteralValue, **payload: Any):
        return self.noop()

    def tensor(self, abstract: AbstractTensor, **payload: Any):
        return self.noop()

    def index(self, item: AbstractIndex, bound: bool, kron: bool,
              **payload: Any):
        return self.noop()

    def indices(self, items: AbstractIndex, **payload: Any):
        return self.noop()

    def index_notation(
        self, tensor: Any, indices: Iterable[Any], **payload: Any
    ):
        return self.noop()

    def call(self, f: str, x: Any, **payload: Any):
        return self.noop()

    def pow(self, base: Any, exponent: Any, **payload: Any):
        return self.noop()

    def neg(self, x: Any, **payload: Any):
        return self.noop()

    def ein(self, lhs: Any, rhs: Any, precedence: int, reduction: str,
            pairwise: str, outidx: Any, **payload: Any):
        return self.noop()

    def tran(self, src: Any, indices: Iterable[Any]):
        return self.noop()

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
    Tensorial = Union[
        P.index_notation, P.call, P.pow, P.neg, P.ein
    ]
    Numeric = Union[Tensorial, Real]

    def as_payload(*k):
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

    @abstractmethod
    def noop(self):
        '''wildcard method for no-operations.'''

    def literal(self, value: LiteralValue, **payload: Any):
        return self.noop()

    def tensor(self, abstract: AbstractTensor, **payload: Any):
        return self.noop()

    def index(self, item: AbstractIndex, bound: bool, kron: bool,
              **payload: Any):
        return self.noop()

    def indices(self, items: Tuple[AbstractIndex], **payload: Any):
        return self.noop()

    def index_notation(
        self, tensor: P.tensor, indices: P.indices, **payload: Any
    ):
        return self.noop()

    def call(self, f: str, x: Tensorial, **payload: Any):
        return self.noop()

    def pow(self, base: Numeric, exponent: Numeric, **payload: Any):
        return self.noop()

    def neg(self, x: Numeric, **payload: Any):
        return self.noop()

    def ein(self, lhs: Numeric, rhs: Numeric, precedence: int, reduction: str,
            pairwise: str, outidx: Optional[P.indices], **payload: Any):
        return self.noop()

    def tran(self, src: Numeric, indices: P.indices):
        return self.noop()

    def __call__(self, node: _ASNode, parent: _ASNode = None):
        node = copy.copy(node)
        for name, value in node.fields_fixed.items():
            setattr(node, name, _deep_apply(self, value, node))
        rule = getattr(self, node.name)
        payload = rule(**node.fields)
        if isinstance(payload, dict):
            node.__dict__.update(**payload)
        elif isinstance(payload, list):
            for key, value in payload:
                setattr(node, key, value)
        elif isinstance(payload, tuple) and len(payload) == 2:
            setattr(node, *payload)
        else:
            raise TypeError(f'Unrecognizable type for payload {payload}')
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
