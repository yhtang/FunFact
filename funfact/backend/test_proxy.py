#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest
import os
from . import (
    active_backend,
    available_backends,
    use,
)


def test_available_backends():
    for backend in available_backends:
        assert os.path.exists(
            os.path.join(
                os.path.dirname(__file__),
                f'_{backend}.py'
            )
        )


def test_use():
    use('numpy')
    assert 'NumPy' in repr(active_backend)

    with pytest.raises(ModuleNotFoundError):
        use('non-existing backend')


def test_active_backend():
    assert repr(active_backend) is not None
    assert repr(active_backend) != 'None'
