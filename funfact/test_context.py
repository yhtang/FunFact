#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest  # noqa: F401
from unittest.mock import MagicMock
import funfact.backend
from .context import is_grad_on, enable_grad


def test_is_grad_on():
    assert isinstance(is_grad_on(), bool)


def test_enable_grad():
    with enable_grad(True):
        assert is_grad_on() is True
        with enable_grad(False):
            assert is_grad_on() is False
        assert is_grad_on() is True


def test_enable_grad_torch():

    mock_backend = MagicMock(
        __name__='Torch'
    )

    prev_backend = getattr(funfact.backend, '_active_backend')
    setattr(funfact.backend, '_active_backend', mock_backend)

    with enable_grad(True):
        mock_backend.set_grad_enabled.assert_called_with(True)
        assert is_grad_on() is True
        with enable_grad(False):
            mock_backend.set_grad_enabled.assert_called_with(False)
            assert is_grad_on() is False
        assert is_grad_on() is True

    setattr(funfact.backend, '_active_backend', prev_backend)
