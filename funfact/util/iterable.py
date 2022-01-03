#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import namedtuple
from typing import List, Tuple


def flatten(iterable):
    '''Converts a nested iterable sequence of iterables and/or non-iterable
    elements into a flat tuple in depth-first order.

    Parameters
    ----------
    iterable: iterable
        The iterable to be flattened

    Returns
    -------
    flat: tuple
        A flattened tuple.
    '''
    def _flatten(_iterable):
        for item in _iterable:
            try:
                yield from _flatten(item)
            except TypeError:
                yield item
    return tuple(_flatten(iterable))


def flatten_if(iterable, pred):
    '''Converts a nested iterable sequence of into a flat tuple in depth-first
    order. Only flattens items recursively if ``pred`` evaluates to True for
    the item.

    Parameters
    ----------
    iterable: iterable
        The iterable to be flattened
    pred: callable
        A callable that returns True for elements that requires further
        flattening.

    Returns
    -------
    flat: tuple
        A flattened tuple.

    '''
    def _flatten_if(_iterable):
        for item in _iterable:
            if pred(item):
                yield from _flatten_if(item)
            else:
                yield item
    return tuple(_flatten_if(iterable))


def flatten_dict(iterable):
    '''Converts a nested dictionary of dictionaries and/or non-iterable
    elements into a flat tuple in depth-first order.

    Parameters
    ----------
    iterable: dict
        The dictionary to be flattened

    Returns
    -------
    flat: tuple
        A flattened tuple.
    '''
    def _flatten_dict(_iterable):
        for _, item in _iterable.items():
            if isinstance(item, dict):
                yield from _flatten_dict(item)
            else:
                yield item
    return tuple(_flatten_dict(iterable))


def map_or_call(iterable, mapping):
    '''Apply ``mapping`` to each element of ``iterable`` using either key-based
    lookup or function evaluation.
    '''
    for item in iterable:
        try:
            yield mapping[item]
        except TypeError:
            yield mapping(item)


def as_namedtuple(title, **kwargs):
    return namedtuple(title, kwargs.keys())(*kwargs.values())


def as_tuple(elements):
    if isinstance(elements, (List, Tuple)):
        return tuple(elements)
    else:
        return (elements,)
