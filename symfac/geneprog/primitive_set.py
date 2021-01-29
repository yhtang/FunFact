#!/usr/bin/env python
# -*- coding: utf-8 -*-
import uuid
import inspect
from abc import ABC, abstractmethod
import numpy as np
from deap import gp
from ntf.util.iterable import flatten, flatten_dict, map_or_call


class PrimitiveSet:
    '''A DEAP primitive set for nonlinear tensor factorization.

    Parameters
    ----------
    ret_type: type
        Type of the overall factorization expression.
    '''

    @staticmethod
    def new_type(name=None, bases=()):
        '''A utility to simplify the creation of abstract types. These abstract
        types can be used to define the input and output of strong-typed GP
        primitives.

        Parameters
        ----------
        name (optional): str
            Name of the type. If None, a random identifier will be assigned.
        bases (optional): tuple
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

    def from_string(self, string):
        '''Convert an expression in the form of a character string into a
        tree of primitives using the current primitive set.

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
        '''Propose a candidate nonlinear factorization expression.

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

    def instantiate(self, expr, **hyper_params):
        '''Create a concrete matrix factorization using the given expression.

        Parameters
        ----------
        expr: list or :py:class:`gp.PrimitiveTree`
            An expression representing a symbolic factorization.
        hyper_params: keyword arguments
            Hyperparameters to be forwarded to the primitives.

        Returns
        -------
        f: callable
            An object that when evaluated returns a reconstructed matrix from
            the given factorization.
        '''
        try:
            return self._instantiate_LL1(expr, **hyper_params)[0]
        except Exception as e:
            raise RuntimeError(
                f'When instantiating the expression {expr}, the following '
                f'exception occurred:\n\n{e}'
            )

    def _instantiate_LL1(self, expr, **hyper_params):
        primitive, tail_expr = expr[0], expr[1:]
        children = []
        for _ in range(primitive.arity):
            child, tail_expr = self._instantiate_LL1(tail_expr, **hyper_params)
            children.append(child)
        primitive_impl = self.pset.context[primitive.name]
        return primitive_impl(*children, **hyper_params), tail_expr

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

            class Primitive:

                def __init__(self, *children, **kwargs):
                    self.__f = f(self, **self._make_hargs(kwargs))
                    self.__c = children

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
                                f'Hyperparameter {k} of primitive {self.name} '
                                f'not provided.'
                            )
                    return hargs

                def __call__(self):
                    return self.__f(*[c() for c in self.__c])

                @property
                def name(self):
                    return _name

                @property
                def unique_name(self):
                    return f'{self.name}_{id(self):x}'

                @property
                def children(self):
                    return self.__c

                def dparam(self, deep=False):
                    if deep is True:
                        return dict(
                            **{p: getattr(self, p) for p in _params},
                            **{c.unique_name: c.dparam(True)
                               for c in self.children}
                        )
                    else:
                        return {p: getattr(self, p) for p in _params}

                @property
                def parameters(self):
                    return flatten_dict(self.dparam(deep=True))

                @property
                def hyperparams(self):
                    return _hyperparams

                @property
                def hyperdefaults(self):
                    return _hyperdefaults

            if in_types is None:
                self.pset.addTerminal(
                    Primitive, ret_type, name=_name
                )
            else:
                self.pset.addPrimitive(
                    Primitive, in_types, ret_type, name=_name
                )

        return decorator

    def add_terminal(self, ret_type, name=None, params=None):
        return self.add_primitive(
            ret_type, in_types=None, name=name, params=params
        )
