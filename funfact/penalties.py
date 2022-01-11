#!/usr/bin/env python
# -*- coding: utf-8 -*-
from funfact import active_backend as ab


class UpperTriangular:
    '''Evaluates upper triangular penalty.'''

    def __call__(self, data):
        return ab.square(ab.abs(ab.tril(data, -1))).mean()


is_upper_triangular = UpperTriangular()
