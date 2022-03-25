#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tqdm
from funfact.backend import active_backend as ab
from ._base import GradientDescentState, GradientDescentPlugin


def gradient_descent(g, optimizer, max_steps, exit_condition, plugins):

    for step in tqdm.trange(max_steps):

        loss, grad = g()

        state = GradientDescentState(
            step=step,
            loss=loss,
            grad=grad,
        )

        with ab.no_grad():

            optimizer.step(grad)

            for plugin in plugins:
                if plugin.trigger(state):
                    plugin.action(state)

        if exit_condition(state):
            break
