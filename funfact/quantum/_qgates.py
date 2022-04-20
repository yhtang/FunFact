#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from numbers import Integral
from typing import List
from funfact import active_backend as ab
import numpy as np
from funfact.parametrized import Generator
from funfact.lang._tsrex import TsrEx, tensor
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import ParametrizedAbstractTensor
from funfact.lang._special import eye
from ._special import proj0, proj1
from funfact.conditions import Unitary
# TODO: how to handle dtype? Should always be complex!


class Gate(ABC):
    '''A common base class for all gate objects.'''
    def __init__(self, qubits: List[Integral]):
        '''A gate stores the minimum and maximum qubits it acts on.'''
        for i, q in enumerate(qubits):
            if not (isinstance(q, Integral) and q >= 0):
                raise RuntimeError(
                    f"Qubit indices must be non-negative integers, got {q} for"
                    f" entry {i}."
                )
        qubits.sort()
        self._qubits = qubits

    @abstractmethod
    def to_tensor(self):
        '''A gate can be converted to a tensor, either parametrized or not.'''
        pass

    @abstractmethod
    def label(self):
        '''A gate has a label.'''
        pass

    @property
    def qubits(self):
        return self._qubits


class OneQubitGate(Gate):
    '''Abstract base class for gates acting on exactly one qubit.'''
    def __init__(self, qubit: Integral):
        if not (isinstance(qubit, Integral)):
            raise RuntimeError(
                f"Qubit index of one qubit gate must be integer, got {qubit}."
            )
        super().__init__(list((qubit,)))


class OneQubitUnitary(OneQubitGate):
    '''A dense one qubit unitary gate.'''
    def __init__(self, qubit: int, initializer=None):
        self._initializer = initializer
        super().__init__(qubit)

    def to_tensor(self):
        return tensor(
            self.label, 2, 2, initializer=self._initializer, prefer=Unitary()
        )

    @property
    def label(self):
        return 'Q_1'


class PauliX(OneQubitGate):
    '''A one qubit Pauli-X gate.'''
    def to_tensor(self):
        return tensor(self.label, np.array([[0, 1], [1, 0]]))

    @property
    def label(self):
        return 'X'


class PauliY(OneQubitGate):
    '''A one qubit Pauli-Y gate.'''
    def to_tensor(self):
        return tensor(self.label, np.array([[0, -1j], [1j, 0]]))

    @property
    def label(self):
        return 'Y'


class PauliZ(OneQubitGate):
    '''A one qubit Pauli-Z gate.'''
    def to_tensor(self):
        return tensor(self.label, np.array([[1, 0], [0, -1]]))

    @property
    def label(self):
        return 'Z'


class OneQubitRotationGate(OneQubitGate):
    '''A common base class for all one qubit Pauli rotations.'''
    def __init__(self, qubit, initializer=None, optimizable=None):
        super().__init__(qubit)
        self.initializer = initializer
        self.optimizable = optimizable

    def _generator(self, theta):
        '''Generates the rotation matrix.'''
        pass

    def to_tensor(self):
        return TsrEx(
            P.parametrized_tensor(
                ParametrizedAbstractTensor(
                    Generator(self._generator, 1), 2, 2, symbol=self.label,
                    initializer=self.initializer, optimizable=self.optimizable
                )
            )
        )


class RotationX(OneQubitRotationGate):
    '''A one qubit rotation along the Pauli-X axis.'''
    def _generator(self, theta, slices=None):
        return ab.vstack(
            [ab.hstack([ab.cos(theta/2), -1j*ab.sin(theta/2)]),
             ab.hstack([-1j*ab.sin(theta/2), ab.cos(theta/2)])]
        )

    @property
    def label(self):
        return 'RX'


class RotationY(OneQubitRotationGate):
    '''A one qubit rotation along the Pauli-Y axis.'''
    def _generator(self, theta, slices=None):
        return ab.vstack(
            [ab.hstack([ab.cos(theta/2), -ab.sin(theta/2)]),
             ab.hstack([ab.sin(theta/2), ab.cos(theta/2)])]
        )

    @property
    def label(self):
        return 'RY'


class RotationZ(OneQubitRotationGate):
    '''A one qubit rotation along the Pauli-Z axis.'''
    def _generator(self, theta, slices=None):
        return ab.vstack(
            [ab.hstack([ab.exp(-1j*theta/2), 0]),
             ab.hstack([0, ab.exp(1j*theta/2)])]
        )

    @property
    def label(self):
        return 'RZ'


class ControlledOneQubitGate(Gate):
    '''Abstract base class for one qubit gate with one control qubit.'''
    def __init__(self, control: Integral, target: Integral):
        if (control == target):
            raise RuntimeError(
                f'control and target qubit must differ, got {control} and '
                f'{target}.'
            )
        self._control = control
        self._target = target
        super().__init__([control, target])

    def to_tensor(self):
        if self._control < self._target:
            return (
                    proj0() &
                    eye(2**(self._target - self._control))
                   ) + (
                    proj1() &
                    eye(2**(self._target - self._control - 1)) &
                    self._gate.to_tensor()
                   )
        return (
                eye(2**(self._control - self._target)) &
                proj0()
               ) + (
                self._gate.to_tensor() &
                eye(2**(self._control - self._target - 1)) &
                proj1()
               )

    @property
    def label(self):
        return self._gate.label


class CX(ControlledOneQubitGate):
    '''A controlled-X (CNOT) gate.'''
    def __init__(self, control, target):
        super().__init__(control, target)
        self._gate = PauliX(target)


class TwoQubitUnitary(Gate):
    '''A dense two qubit unitary gate.'''
    def __init__(self, qubit, initializer=None):
        '''The two qubit unitary acts on two consecutive qubits:
        qubit, qubit + 1.'''
        if not (isinstance(qubit, Integral) and qubit >= 0):
            raise RuntimeError(
                f"Qubit index must be non-negative integer, got {qubit}."
                )
        self._initializer = initializer
        super().__init__([qubit, qubit+1])

    def to_tensor(self):
        return tensor(
            self.label, 4, 4, initializer=self._initializer, prefer=Unitary()
        )

    @property
    def label(self):
        return 'Q_2'
