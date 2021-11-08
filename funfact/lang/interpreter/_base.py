#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import copy
from numbers import Real
from typing import Any, Callable, Iterable, Union
from funfact.lang._ast import _ASNode, _AST, Primitives as P
from funfact.lang._tensor import AbstractIndex, AbstractTensor
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
    def scalar(self, value: Real, **payload: Any):
        pass

    @abstractmethod
    def tensor(self, value: AbstractTensor, **payload: Any):
        pass

    @abstractmethod
    def index(self, value: AbstractIndex, **payload: Any):
        pass

    @abstractmethod
    def index_notation(
        self, tensor: Any, indices: Iterable[Any], **payload: Any
    ):
        pass

    @abstractmethod
    def call(self, f: str, x: Any, **payload: Any):
        pass

    @abstractmethod
    def pow(self, base: Any, exponent: Any, **payload: Any):
        pass

    @abstractmethod
    def neg(self, x: Any, **payload: Any):
        pass

    @abstractmethod
    def mul(self, lhs: Any, rhs: Any, **payload: Any):
        pass

    @abstractmethod
    def div(self, lhs: Any, rhs: Any, **payload: Any):
        pass

    @abstractmethod
    def add(self, lhs: Any, rhs: Any, **payload: Any):
        pass

    @abstractmethod
    def sub(self, lhs: Any, rhs: Any, **payload: Any):
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
    Tensorial = Union[
        P.index_notation, P.call, P.pow, P.neg, P.mul, P.div, P.add, P.sub
    ]
    Numeric = Union[Tensorial, Real]

    def as_payload(k):
        def wrapper(f):
            def wrapped_f(*args, **kwargs):
                return k, f(*args, **kwargs)
            return wrapped_f
        return wrapper

    @abstractmethod
    def scalar(self, value: Real, **payload: Any):
        pass

    @abstractmethod
    def tensor(self, value: AbstractTensor, **payload: Any):
        pass

    @abstractmethod
    def index(self, value: AbstractIndex, **payload: Any):
        pass

    @abstractmethod
    def index_notation(
        self, tensor: P.tensor, indices: Iterable[P.index], **payload: Any
    ):
        pass

    @abstractmethod
    def call(self, f: str, x: Tensorial, **payload: Any):
        pass

    @abstractmethod
    def pow(self, base: Numeric, exponent: Numeric, **payload: Any):
        pass

    @abstractmethod
    def neg(self, x: Numeric, **payload: Any):
        pass

    @abstractmethod
    def mul(self, lhs: Numeric, rhs: Numeric, **payload: Any):
        pass

    @abstractmethod
    def div(self, lhs: Numeric, rhs: Numeric, **payload: Any):
        pass

    @abstractmethod
    def add(self, lhs: Numeric, rhs: Numeric, **payload: Any):
        pass

    @abstractmethod
    def sub(self, lhs: Numeric, rhs: Numeric, **payload: Any):
        pass

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
            raise TypeError(f'Uncognizable type for payload {payload}')
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


def dfs_filter(function: Callable[[_ASNode], bool], node: _ASNode):
    '''Returns an iterator that loop over all nodes in an AST in a depth-first
    manner for which `function` evaluates to trues.'''

    for child in flatten_if(
        node.fields_fixed.values(),
        lambda elem: isinstance(elem, (list, tuple))
    ):
        if isinstance(child, _ASNode):
            yield from dfs_filter(function, child)

    if function(node) is True:
        yield node