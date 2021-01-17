#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import sympy as sy
from sympy.printing.pycode import PythonCodePrinter
from sympy.utilities.lambdify import lambdify
import torch
from itertools import chain

_known_functions_torch = dict(
    exp='exp',
    sin='sin',
    cos='cos',
    outer='outer'
)

_known_constants_numpy = {
    'Exp1': 'e',
    'Pi': 'pi',
    'EulerGamma': 'euler_gamma',
    'NaN': 'nan',
    'Infinity': 'PINF',
    'NegativeInfinity': 'NINF'
}

class TorchPrinter(PythonCodePrinter):
    printmethod = "_torchcode"
    language = "Python with PyTorch and Numpy"

    _kf = dict(chain(
        PythonCodePrinter._kf.items(),
        [(k, 'torch.' + v) for k, v in _known_functions_torch.items()]
    ))
    _kc = {k: 'numpy.'+v for k, v in _known_constants_numpy.items()}

    def _print_MatPow(self, expr):
        "Matrix power printer"
        return '{}({}, {})'.format(self._module_format('torch.matrix_power'),
            self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_ZeroMatrix(self, expr):
        return '{}({})'.format(self._module_format('torch.zeros'),
            self._print(expr.shape))

    def _print_OneMatrix(self, expr):
        return '{}({})'.format(self._module_format('torch.ones'),
            self._print(expr.shape))

    def _print_HadamardProduct(self, expr):
        func = self._module_format('torch.mul')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_KroneckerProduct(self, expr):
        func = self._module_format('torch.kron')
        return ''.join('{}({}, '.format(func, self._print(arg)) \
            for arg in expr.args[:-1]) + "{}{}".format(self._print(expr.args[-1]),
            ')' * (len(expr.args) - 1))

    def _print_DiagMatrix(self, expr):
        return '{}({})'.format(self._module_format('torch.diag'),
            self._print(expr.args[0]))

    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        op = {
            '==': 'eq',
            '!=': 'ne',
            '<' : 'lt',
            '<=': 'le',
            '>' : 'gt',
            '>=': 'ge',
        }
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '{op}({lhs}, {rhs})'.format(op=self._module_format('torch.'+op[expr.rel_op]),
                                               lhs=lhs, rhs=rhs)
        return super()._print_Relational(expr)

    def _print_And(self, expr):
        "Logical And printer"
        # We have to override LambdaPrinter because it uses Python 'and' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_and' to NUMPY_TRANSLATIONS.
        return '{}.reduce(({}))'.format(self._module_format('torch.logical_and'), ','.join(self._print(i) for i in expr.args))

    def _print_Or(self, expr):
        "Logical Or printer"
        # We have to override LambdaPrinter because it uses Python 'or' keyword.
        # If LambdaPrinter didn't define it, we could use StrPrinter's
        # version of the function and add 'logical_or' to NUMPY_TRANSLATIONS.
        return '{}.reduce(({}))'.format(self._module_format('torch.logical_or'), ','.join(self._print(i) for i in expr.args))

    def _print_Not(self, expr):
        "Logical Not printer"
        # We have to override LambdaPrinter because it uses Python 'not' keyword.
        # If LambdaPrinter didn't define it, we would still have to define our
        #     own because StrPrinter doesn't define it.
        return '{}({})'.format(self._module_format('torch.logical_not'), ','.join(self._print(i) for i in expr.args))

    def _print_Pow(self, expr, rational=False):
        # XXX Workaround for negative integer power error
        from sympy.core.power import Pow
        if expr.exp.is_integer and expr.exp.is_negative:
            expr = Pow(expr.base, expr.exp.evalf(), evaluate=False)
        return self._hprint_Pow(expr, rational=rational, sqrt='torch.sqrt')

    def _print_Min(self, expr):
        return '{}(({}), axis=0)'.format(self._module_format('torch.amin'), ','.join(self._print(i) for i in expr.args))

    def _print_Max(self, expr):
        return '{}(({}), axis=0)'.format(self._module_format('torch.amax'), ','.join(self._print(i) for i in expr.args))

    def _print_arg(self, expr):
        return "%s(%s)" % (self._module_format('torch.angle'), self._print(expr.args[0]))

    def _print_re(self, expr):
        return "%s(%s)" % (self._module_format('torch.real'), self._print(expr.args[0]))

    def _print_im(self, expr):
        return "%s(%s)" % (self._module_format('torch.imag'), self._print(expr.args[0]))

    def _print_Mod(self, expr):
        return "%s(%s)" % (self._module_format('torch.fmod'), ', '.join(
            map(lambda arg: self._print(arg), expr.args)))

    def _print_Identity(self, expr):
        shape = expr.shape
        if all([dim.is_Integer for dim in shape]):
            return "%s(%s)" % (self._module_format('torch.eye'), self._print(expr.shape[0]))
        else:
            raise NotImplementedError("Symbolic matrix dimensions are not yet supported for identity matrices")

    def _print_CodegenArrayElementwiseAdd(self, expr):
        return self._expand_fold_binary_op('torch.add', expr.args)

def _print_known_func(self, expr):
    known = self.known_functions[expr.__class__.__name__]
    return '{name}({args})'.format(name=self._module_format(known),
                                   args=', '.join(map(lambda arg: self._print(arg), expr.args)))


def _print_known_const(self, expr):
    known = self.known_constants[expr.__class__.__name__]
    return self._module_format(known)
    
for k in TorchPrinter._kf:
    setattr(TorchPrinter, '_print_%s' % k, _print_known_func)

for k in TorchPrinter._kc:
    setattr(TorchPrinter, '_print_%s' % k, _print_known_const)
