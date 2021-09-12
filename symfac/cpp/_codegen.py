#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import os
import copy
from contextlib import contextmanager


class Template:
    """A template string with placeholders in the format of ${...} or ?{...},
    which can be rendered on-demand. Helpful for code generation.

    Parameters
    ----------
    template: str
        A template string containing placeholders in the format of ${...}.
    escape: bool
        Whether or not backslash (\\) should be escaped in the fill-in contents
        to the placeholders.
    """

    def __init__(self, template, escape=True):
        if os.path.isfile(template):
            self.template = open(template).read()
        else:
            self.template = template
        self.escape = escape

    @contextmanager
    def context(self, **kwargs):
        t = copy.copy(self)
        t.template = re.sub(
            r'\?{([^}]+)}',
            lambda m: 'true' if eval(m.group(1), kwargs) else 'false',
            t.template
        )
        yield t

    def render(self, **substitutions):
        """Substitute placeholders using the syntax symbol=replacement.
        Partial renderings are allowed. If replacement is list-like, use the
        trailing sequence of symbol match in placeholders to join members;
        otherwise do plain substitution.
        """

        text = self.template
        for symbol in sorted(substitutions, key=lambda s: (-len(s), s)):
            repl = substitutions[symbol]
            if isinstance(repl, (list, tuple)):
                pattern = r'\${%s([^}]*)}' % symbol
                repl = [str(r) for r in repl]
                text = re.sub(pattern, lambda m: m.group(1).join(repl), text)
            else:
                pattern = r'\${%s}' % symbol
                repl = str(repl)
                if self.escape is False:
                    repl = repl.replace('\\', r'\\')
                text = re.sub(pattern, repl, text)
        return text
