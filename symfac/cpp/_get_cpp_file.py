#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


def get_cpp_file(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()
