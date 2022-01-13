#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._base import ROOFInterpreter


_omap = dict(
    sum=r'\sum',
    negative='-',
    conj=r'\operatorname{conj}',
    add='+',
    subtract='-',
    multiply=r'',
    divide='/',
    float_power='^',
    matmul='',
    kron=r'\otimes',
    min=r'\min',
    max=r'\max',
    log_sum_exp='LSE',
    log_add_exp='LAE'
)


class LatexRenderer(ROOFInterpreter):

    def __call__(self, node, parent=None):
        '''Decorate the base evaluation result with an optional pair of
        parentheses conditional on the relative precedence between the parent
        and child nodes.'''
        value = super().__call__(node, parent)
        if parent is not None and node.precedence > parent.precedence:
            return fr'\left({value}\right)'
        else:
            return value

    def abstract_index_notation(self, tensor, indices, **kwargs):
        return fr'''{{{tensor}}}_{{{indices}}}'''

    def abstract_binary(self, lhs, rhs, precedence, operator, **kwargs):
        return fr'{{{lhs}}} {_omap[operator]} {{{rhs}}}'

    def literal(self, value, **kwargs):
        return value._repr_tex_()

    def tensor(self, decl, **kwargs):
        return decl._repr_tex_()

    def index(self, item, bound, kron, **kwargs):
        if bound:
            accent = r'\widetilde'
        elif kron:
            accent = r'\overset{\otimes}'
        else:
            accent = None
        return fr'{{{item._repr_tex_(accent=accent)}}}'

    def indices(self, items, **kwargs):
        return ''.join(items)

    def indexed_tensor(self, tensor, indices, **kwargs):
        return fr'''{{{tensor}}}_{{{indices}}}'''

    def call(self, f, x, **kwargs):
        return fr'\operatorname{{{f}}}{{{x}}}'

    def neg(self, x, **kwargs):
        return fr'-{x}'

    def elem(self, lhs, rhs, precedence, operator, **kwargs):
        return fr'{{{lhs}}} {_omap[operator]} {{{rhs}}}'

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, **kwargs):
        if reduction == 'sum':
            op = _omap[pairwise]
        else:
            op = r'\underset{{{}:{}}}{{\star}}'.format(
                _omap[reduction], _omap[pairwise]
            )
        tex = fr'{{{lhs}}} {op} {{{rhs}}}'
        if outidx is not None:
            tex = fr'({tex})\rightarrow_{{{outidx}}}'
        return tex

    def tran(self, src, indices, **kwargs):
        return fr'{{{src}}}\rightarrow_{{{indices}}}'

    def abstract_dest(self, src, indices, **kwargs):
        return fr'{{{src}}}\rightarrow_{{{indices}}}'
