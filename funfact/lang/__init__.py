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
            "<script src='https://www.gstatic.com/external_hosted/"
            "mathjax/latest/MathJax.js?config=default'></script>"
        ))
    )
except Exception:
    warnings.warn('Cannot set up MathJAX, LaTeX rendering may not work.')


__all__ = ['index', 'indices', 'tensor', 'template', '_0', '_1', 'delta']
