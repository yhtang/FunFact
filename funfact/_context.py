#!/usr/bin/env python
# -*- coding: utf-8 -*-
import contextlib
import enum


class _EagerMode:

    class _Eagerness(enum.Enum):
        LAZY = -1
        DEFAULT = 0
        EAGER = 1

    current = _Eagerness.DEFAULT
    previous = []

    @classmethod
    def _push(cls):
        cls.previous.append(cls.current)

    @classmethod
    def _pop(cls):
        cls.current = cls.previous.pop()


def set_eagerness(mode):
    _EagerMode.current = _EagerMode._Eagerness(mode)


def get_eagerness():
    return _EagerMode.current


def push_eagerness():
    _EagerMode._push()


def pop_eagerness():
    _EagerMode._pop()


@contextlib.contextmanager
def eager_mode():
    try:
        push_eagerness()
        set_eagerness(_EagerMode._Eagerness.EAGER)
        yield
    finally:
        pop_eagerness()


@contextlib.contextmanager
def lazy_mode():
    try:
        push_eagerness()
        set_eagerness(_EagerMode._Eagerness.LAZY)
        yield
    finally:
        pop_eagerness()


@contextlib.contextmanager
def default_mode():
    try:
        push_eagerness()
        set_eagerness(_EagerMode._Eagerness.DEFAULT)
        yield
    finally:
        pop_eagerness()
