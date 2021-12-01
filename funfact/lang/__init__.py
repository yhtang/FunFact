#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from ._tsrex import index, indices, tensor
from ._tplex import template
from ._predefined_literal import _0, _1, delta


try:
    from IPython import get_ipython
    from IPython.display import display, HTML

    get_ipython().events.register(
        'pre_run_cell',
        lambda: display(HTML(
            "<script type='text/javascript' async "
            "src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/"
            "MathJax.js?config=TeX-MML-AM_CHTML'></script>"
        ))
    )
except Exception:
    warnings.warn('Cannot set up MathJAX, LaTeX rendering may not work.')


__all__ = ['index', 'indices', 'tensor', 'template', '_0', '_1', 'delta']
