#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


def get_cpp_file(*path):
    return open(os.path.join(os.path.dirname(__file__), *path)).read()
