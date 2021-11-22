#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ._base import ROOFInterpreter


_omap = dict(
    add='+',
    sub='-',
    mul=r'\times',
    div='/',
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

    def literal(self, value, **kwargs):
        return value._repr_tex_()

    def tensor(self, abstract, **kwargs):
        return abstract._repr_tex_()

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

    def index_notation(self, tensor, indices, **kwargs):
        return fr'''{{{tensor}}}_{{{indices}}}'''

    def call(self, f, x, **kwargs):
        return fr'\operatorname{{{f}}}{{{x}}}'

    def pow(self, base, exponent, **kwargs):
        return fr'{{{base}}}^{{{exponent}}}'

    def neg(self, x, **kwargs):
        return fr'-{x}'

    def ein(self, lhs, rhs, precedence, reduction, pairwise, outidx, **kwargs):
        if reduction == 'sum':
            op = _omap[pairwise]
        else:
            op = r'\underset{{{}:{}}}{{\star}}'.format(
                _omap[reduction], _omap[pairwise]
            )
        body = fr'{lhs} {op} {rhs}'
        suffix = fr'\rightarrow_{{{outidx}}}' if outidx is not None else ''
        return body + suffix

    def tran(self, src, indices, **kwargs):
        return fr'{{{src}}}^{{\mathsf{{T}}: {indices}}}'
