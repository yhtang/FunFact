#!/usr/bin/env python
# -*- coding: utf-8 -*-
import uuid
import inspect
from abc import ABC, abstractmethod
import numpy as np
from deap import gp
from funfact.util.iterable import flatten, map_or_call
from .factorization import Factorization


# TODO:
# To be rewritten: factor out the action/interpretation part and only keeps the
# representation part. The former is then going into the evaluation module.


class Primitive(ABC):
    '''A primitive in the context of genetic programming is an element of an
    arithmetic expression, such as a constant, an operation, a function, etc.
    '''

    def __repr__(self):
        return f'<{self.name} object #{id(self):x}>'

    def _make_hargs(self, kwargs):
        hargs = {}
        for k in self.hyperparams:
            if k in kwargs:
                hargs[k] = kwargs.pop(k)
            elif k in self.hyperdefaults:
                hargs[k] = self.hyperdefaults[k]
            else:
                raise RuntimeError(
                    f'Hyperparameter `{k}` of primitive `{self.name}` '
                    f'not provided.'
                )
        return hargs

    @property
    @abstractmethod
    def name(self):
        '''The primitive's name as it shows up in an abstract expression.'''

    @property
    def unique_name(self):
        '''The name of the primitive as it shows up in a concrete
        instance of an abstract expression. It consists of the name of the
        primitive plus a unique ID in order to distinguish instances of the
        same type.'''
        return f'{self.name}_{id(self):x}'

    @property
    @abstractmethod
    def action(self):
        '''A callable that carries out the action of the primitive.'''

    @property
    @abstractmethod
    def parameter_name(self):
        '''The names of the optimizable paramters of the primitive.'''

    @property
    def parameters(self):
        return {p: getattr(self, p) for p in self.parameter_name}

    @property
    @abstractmethod
    def hyperparams(self):
        '''The names of the hyperparamters of the primitive.'''

    @property
    @abstractmethod
    def hyperdefaults(self):
        '''The default values to the hyperparamters of the primitive.'''


class PrimitiveSet:
    '''A primitive set, i.e. a realization of a factorzation context-free
    grammar (CFG), for nonlinear tensor factorization. This is built on top
    of the `PrimitiveSetTyped` concept of the DEAP package.

    Parameters
    ----------
    ret_type: type
        The grammatical type of the overall factorization expression.
    '''

    @staticmethod
    def new_type(name=None, bases=()):
        '''A utility to simplify the creation of grammatical types. These
        abstract types can be used to define the input and output of
        strong-typed GP primitives.

        Parameters
        ----------
        name: str
            Name of the type. If None, a random identifier will be assigned.
        bases: tuple
            A tuple of base classes that the new type will inherit from.

        Returns
        -------
        t: type
            A new type.
        '''
        name = name or f'type{uuid.uuid4().hex}'
        return type(name, bases, {})

    def __init__(self, ret_type):
        self.ret_type = ret_type
        self.pset = gp.PrimitiveSetTyped('factorization', [], ret_type)
        self.hyperdep = {}

    @property
    def primitives(self):
        return self.pset.primitives

    @property
    def terminals(self):
        return self.pset.terminals

    @property
    def context(self):
        return self.pset.context

    def from_string(self, string):
        '''Converts a character string into a tree of primitives using the
        current primitive set.

        Parameters
        ----------
        string: str
            A character string such as 'add(x, y)'.

        Returns
        -------
        expr: :py:class:`gp.PrimitiveTree`
            A tree representation of the primitives. The tree can also be
            iterated over like a list in depth-first order.
        '''
        return gp.PrimitiveTree.from_string(string, self.pset)

    def gen_expr(self, max_depth: int, p=None):
        '''Generate a random nonlinear factorization expression.

        Parameters
        ----------
        max_depth: int
            Maximum depth (number of layers) of the expression.
        p: dict or callable
            A lookup table of the relative frequencies of the primitives in the
            generated expression.

        Returns
        -------
        expr: list
            An abstract factorization in the form of a prefix expression.
        '''
        return self._gen_expr_depth_first(
            self.ret_type,
            p if p is not None else lambda _: 1.0,
            max_depth
        )

    def _gen_expr_depth_first(self, t=None, p=None, d=0):
        if d <= 0:  # try to terminate ASAP
            try:
                choice = np.random.choice(self.pset.terminals[t], 1).item()
            except ValueError:
                choice = np.random.choice(self.pset.primitives[t], 1).item()
        else:  # normal growth
            candidates = self.pset.primitives[t] + self.pset.terminals[t]
            prob = np.fromiter(map_or_call(candidates, p), dtype=np.float)
            choice = np.random.choice(
                candidates, 1, p=prob / prob.sum()
            ).item()

        if isinstance(choice, gp.Terminal):
            return [choice]
        else:
            return [choice, *flatten([self._gen_expr_depth_first(a, p=p, d=d-1)
                                      for a in choice.args])]

    def instantiate(self, expr, **hyperparams):
        '''Create a concrete matrix factorization using the given expression.

        Parameters
        ----------
        expr: list or :py:class:`gp.PrimitiveTree`
            An expression representing a symbolic factorization.
        hyperparams: keyword arguments
            Hyperparameters to be forwarded to the primitives.

        Returns
        -------
        f: :py:class:`Primitive`
            A primitive object, which serves as the root of the abstract syntax
            tree, which when evaluated returns a reconstructed matrix given
            the factorization.
        '''
        try:
            return Factorization(self._instantiate_LL1(expr, **hyperparams)[0])
        except Exception as e:
            raise RuntimeError(
                f'When instantiating the expression {expr}, the following '
                f'exception occurred:\n\n{e}'
            )

    def _instantiate_LL1(self, expr, **hyperparams):
        primitive, tail_expr = expr[0], expr[1:]
        children = []
        for _ in range(primitive.arity):
            child, tail_expr = self._instantiate_LL1(tail_expr, **hyperparams)
            children.append(child)
        prim_def = self.pset.context[primitive.name]
        return (prim_def(**hyperparams), (*children,)), tail_expr

    @staticmethod
    def _get_hyperspecs(f, name):
        arg_spec = inspect.getfullargspec(f)
        assert arg_spec.varargs is None, f'Variable-length hyperparameter \
            *{arg_spec.varargs} is not allowed for primitive {name}.'
        assert arg_spec.varkw is None, f'Variable-length keyword \
            hyperparameter **{arg_spec.varkw} is not allowed for primitive \
            {name}.'
        hyperparams = arg_spec.args[1:] + arg_spec.kwonlyargs

        hyperdefaults = {}
        # defaults for positional arguments
        if arg_spec.defaults is not None:
            for key, value in zip(arg_spec.defaults[-1::-1],
                                  arg_spec.args[-1::-1]):
                hyperdefaults[key] = value
        # defaults for keyword-only arguments
        if arg_spec.kwonlydefaults is not None:
            hyperdefaults.update(**arg_spec.kwonlydefaults)

        return hyperparams, hyperdefaults

    def add_primitive(self, ret_type, in_types=None, name=None, params=None):
        '''A decorator to turn a user-defined function into a non-terminal
        primitive. The user-defined function should perform two tasks: 1)
        initialize the primitive object (given as the first argument to the
        user function) with optimizable parameters, and 2) return a callable
        that accepts the inputs as specified by `in_types` and return an output
        that grammatically conforms to the `ret_type`.

        Parameters
        ----------
        ret_type: type
            The grammatical type of the output of the primitive.
        in_types: list of types
            The grammatical types of the inputs to the primitive.
        name: str
            The name of the primitive as shown in serialized expressions. If
            None, the ``__name__`` attribute of the decorated function will be
            used.
        params: list of strs
            A list of attribute names that corresponds to the optimizable
            parameters of a primitive instance.
            These parameters must be initialized by the decorated user
            function. They will show up in the ``.parameters`` property and
            in the dictionary as returned by ``.dparam()`` method of an
            instantiated expression, and can be conveniently feed into an
            optimizer such as one of those provided by PyTorch.
        '''

        def decorator(f):

            try:
                _name = name or f.__name__
            except AttributeError:
                raise AttributeError(
                    f'Primitive {f} does not have the `__name__` attribute. '
                    f'Please specify one using the `name` argument.'
                )

            _params = params or []
            _hyperparams, _hyperdefaults = self._get_hyperspecs(f, _name)

            for h in _hyperparams:
                if h not in self.hyperdep:
                    self.hyperdep[h] = []
                self.hyperdep[h].append(_name)

            class Prim(Primitive):

                __qualname__ = _name

                def __init__(self, **kwargs):
                    self.__f = f(self, **self._make_hargs(kwargs))

                @property
                def name(self):
                    return _name

                @property
                def action(self):
                    return self.__f

                @property
                def parameter_name(self):
                    return _params

                @property
                def hyperparams(self):
                    return _hyperparams

                @property
                def hyperdefaults(self):
                    return _hyperdefaults

            if in_types is None:
                self.pset.addTerminal(
                    Prim, ret_type, name=_name
                )
            else:
                self.pset.addPrimitive(
                    Prim, in_types, ret_type, name=_name
                )

        return decorator

    def add_terminal(self, ret_type, name=None, params=None):
        '''A decorator to turn a user-defined function into a terminal
        primitive. This is essentially an alias of the
        :py:meth:`add_primitive` method with ``ret_type`` being None.'''
        return self.add_primitive(
            ret_type, in_types=None, name=name, params=params
        )
