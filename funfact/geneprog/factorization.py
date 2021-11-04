#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
from treelib import Tree
import torch
from funfact.util.iterable import flatten_dict

'''
Long-term TODO list:
- [ ] better integration in terms of tree/factorization. A factorization is a
      tree, a tree is a factorization. Intermediate/leaf nodes are primitives,
      subtrees are factorization.
'''


class Factorization:
    '''A factorization object is a concrete realization of a factorizaion
    expression. The parameters of the factorization object can be optimized
    to obtain a better approximation of a target tensor.

    Parameters
    ----------
    tree: treelib.Tree or nested tuple
        A tree of primitives either represented by a :py:class:`treelib.Tree`
        object or a nested tuple containing realizations of primitives and
        their children. The nested tuple can be generated programatically
        using e.g. :py:meth:`PrimitiveSet.instantiate`. As an example,  the
        expression ``g(r0) + g(r1) * h(r2)`` corresponds to the nested tuple:
        .. code-block::
        (add,
         (g, (r0,)),
         (mul,
          (g, (r1,)),
          (h, (r2,))
         )
        )
    '''

    def __init__(self, t):
        if isinstance(t, Tree):
            self.tree = t
        else:
            self.tree = self._as_tree(t)

    @classmethod
    def _as_tree(cls, t):
        primitive, children = t
        tree = Tree()
        r = tree.create_node(
            tag=primitive.name,
            identifier=primitive.unique_name,
            data=primitive
        )
        for c in children:
            tree.paste(r.identifier, cls._as_tree(c))
        return tree

    def __str__(self):
        return self.to_text(key=lambda n: n.data.unique_name)

    def draw(self, label='name', key=None, mode='text'):
        if mode == 'text':
            print(self.to_text(label=label, key=key))

    def to_text(self, label='name', key=None):
        '''Render a ASCII art visualization of the factorization expression.

        Parameters
        ----------
        label: 'name' or 'unique_name'
            The label to be printed to represent each node.
        key: callable
            Returns the sorting key for the nodes.
        '''
        return self.tree.show(
            line_type=u'ascii-ex', key=key, data_property=label, stdout=False
        )

    def __call__(self):
        '''Shorthand for :py:meth:`forward`.'''
        return self.forward()

    def forward(self):
        '''Recursively evaluate the primitive and its children, and return
        the output.'''
        return self.head.action(*[c() for c in self.children])

    @property
    def head(self):
        return self.tree[self.tree.root].data

    @property
    def children(self):
        '''A list of child factorization instances.'''
        return [Factorization(self.tree.subtree(n.identifier))
                for n in self.tree.children(self.tree.root)]

    @property
    def parameters(self):
        '''A flattened list of optimizable parameters of the primitive and all
        its children. It can be directly plugged into a PyTorch optimizer.'''
        return flatten_dict(self.parameters_dict)

    @property
    def parameters_dict(self):
        '''A dictionary of name-value pairs of optimizable
        parameters.'''
        return {
            self.tree[id].data.unique_name: self.tree[id].data.parameters
            for id in self.tree.expand_tree(key=lambda n: n.data.unique_name)
        }

    @property
    def flat_parameters(self):
        with torch.no_grad():
            return torch.cat([p.view(-1) for p in self.parameters])

    @flat_parameters.setter
    def flat_parameters(self, value):
        parameters = self.parameters
        xv = torch.split(value, [p.numel() for p in parameters])
        for v, parameter in zip(xv, parameters):
            parameter.view(-1).data[:] = v

    @staticmethod
    def _match(pattern, target, regex):
        if regex is True:
            return (re.fullmatch(pattern, target) is not None)
        else:
            return target == pattern

    def find(self, pattern, property='name', regex=True, subtree=False):
        '''Return an iterable over all child primitives that matches the
        name.

        Parameters
        ----------
        name: str
            The name or pattern of name for the child primitives to loop over.
        regex: bool
            Whether or not to treat the name as a regular expression.
        subtree: bool
            If True, return the subtree below each found primitive as a
            factorization object. Otherwise, return found primitives as-is.
        '''
        return list(self.ifind(pattern=pattern, property=property,
                               regex=regex, subtree=subtree))

    def ifind(self, pattern, property='name', regex=True, subtree=False):
        for node in self.tree.filter_nodes(
            lambda n: self._match(pattern, getattr(n.data, property), regex),
        ):
            if subtree is True:
                yield type(self)(self.tree.subtree(node.identifier))
            else:
                yield node.data
