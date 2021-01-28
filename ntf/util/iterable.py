#!/usr/bin/env python
# -*- coding: utf-8 -*-


def flatten(iterable):
    def _flatten(_iterable):
        for item in _iterable:
            try:
                yield from _flatten(item)
            except TypeError:
                yield item
    return tuple(_flatten(iterable))


def flatten_if(iterable, pred):
    def _flatten_if(_iterable):
        for item in _iterable:
            if pred(item):
                yield from _flatten_if(item)
            else:
                yield item
    return tuple(_flatten_if(iterable))


def flatten_dict(iterable):
    def _flatten_dict(_iterable):
        for _, item in _iterable.items():
            if isinstance(item, dict):
                yield from _flatten_dict(item)
            else:
                yield item
    return tuple(_flatten_dict(iterable))


def map_or_call(iterable, mapping):
    for item in iterable:
        try:
            yield mapping[item]
        except TypeError:
            yield mapping(item)
