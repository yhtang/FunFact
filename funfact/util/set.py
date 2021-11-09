#!/usr/bin/env python
# -*- coding: utf-8 -*-


def ordered_union(S, T):
    return S + [t for t in T if t not in S]


def ordered_intersect(S, T):
    return [s for s in S if s in T]


def ordered_setminus(S, T):
    return [s for s in S if s not in T]


def ordered_symmdiff(S, T):
    S1 = [s for s in S if s not in T]
    T1 = [t for t in T if t not in S]
    return S1 + T1
