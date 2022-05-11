#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC
from numbers import Integral
from typing import List
from funfact import active_backend as ab
import numpy as np
from funfact.parametrized import Generator
from funfact.lang._tsrex import TsrEx, tensor
from funfact.lang._ast import Primitives as P
from funfact.lang._terminal import ParametrizedAbstractTensor
from funfact.lang._special import eye, proj0, proj1
from funfact.conditions import Unitary


_proj0 = proj0()
_proj1 = proj1()


class _PlacedGate:
    '''A placed gate wraps a gate and contains the qubit position
    information.'''
    def __init__(self, gate, qubits: List[Integral]):
        self._gate = gate
        self._qubits = tuple(qubits)

    def to_tsrex(self):
        return self._gate.to_tsrex(self._qubits)

    @property
    def label(self):
        return self._gate.label

    @property
    def qubits(self):
        '''The minimum and maximum indices of the qubits that the gate acts
        on.'''
        return self._qubits

    def __repr__(self) -> str:
        print(self.label)
        return fr'{self.label}_{{q:{self._qubits}}}'


class Gate(ABC):
    '''A common abstract base class for all gate objects. Gates have a label
    and a tensor expression representation.'''
    def __init__(self):
        pass

    def to_tsrex(self, qubits=None):
        '''Tensor expression representation of the gate.'''
        return self._tsrex

    @property
    def label(self):
        '''A gate has a label.'''
        return self._label

    def __matmul__(self, qubits: List[Integral]):
        if isinstance(qubits, int):
            qubits = list((qubits,))
        return _PlacedGate(self, qubits)

    def __repr__(self) -> str:
        return self.label


class OneQubitUnitary(Gate):
    '''A dense one qubit unitary gate.'''
    def __init__(self, label=None, initializer=None):
        '''
        Args:
            label: string
                Label of the quantum gate.
            initializer
                Initialization distribution
        '''
        self._initializer = initializer
        self._label = label or 'U_1'
        self._tsrex = tensor(
            self._label, 2, 2, initializer=initializer, prefer=Unitary()
        )


class TwoQubitUnitary(Gate):
    '''A dense two qubit unitary gate.'''
    def __init__(self, label=None, initializer=None):
        '''
        Args:
            label: string
                Label of the quantum gate.
            initializer
                Initialization distribution
        '''
        self._initializer = initializer
        self._label = label or 'U_2'
        self._tsrex = tensor(
            self._label, 4, 4, initializer=initializer, prefer=Unitary()
        )


class PauliX(Gate):
    '''A one qubit Pauli-X gate.'''
    _label = 'X'

    def to_tsrex(self, qubits=None):
        return tensor(
            self._label, np.array([[0, 1], [1, 0]]), optimizable=False
        )


class PauliY(Gate):
    '''A one qubit Pauli-Y gate.'''
    _label = 'Y'

    def to_tsrex(self, qubits=None):
        return tensor(
            self._label, np.array([[0, -1j], [1j, 0]]), optimizable=False
        )


class PauliZ(Gate):
    '''A one qubit Pauli-Z gate.'''
    _label = 'Z'

    def to_tsrex(self, qubits=None):
        return tensor(
            self.label, np.array([[1, 0], [0, -1]]), optimizable=False
        )


class OneQubitRotationGate(Gate):
    '''A common base class for all one qubit Pauli rotations.'''
    def __init__(self, initializer=None, optimizable=True):
        '''
        Args:
            initializer:
                Initialization distribution of parameter
            optimizable:
                Flag indicating whether this gate is optimizable or not.
        '''
        self.initializer = initializer
        self.optimizable = optimizable
        self._tsrex = TsrEx(
            P.parametrized_tensor(
                ParametrizedAbstractTensor(
                    Generator(self._generator, 1), 2, 2, symbol=self.label,
                    initializer=self.initializer, optimizable=self.optimizable
                )
            )
        )

    def _generator(self, theta):
        '''Generates the rotation matrix.'''
        pass

    def to_tsrex(self, qubits=None):
        return self._tsrex


class RotationX(OneQubitRotationGate):
    '''A one qubit rotation along the Pauli-X axis.'''
    _label = 'RX'

    def _generator(self, theta, slices=None):
        return ab.vstack(
            [ab.hstack([ab.cos(theta/2), -1j*ab.sin(theta/2)]),
             ab.hstack([-1j*ab.sin(theta/2), ab.cos(theta/2)])]
        )


class RotationY(OneQubitRotationGate):
    '''A one qubit rotation along the Pauli-Y axis.'''
    _label = 'RY'

    def _generator(self, theta, slices=None):
        return ab.vstack(
            [ab.hstack([ab.cos(theta/2), -ab.sin(theta/2)]),
             ab.hstack([ab.sin(theta/2), ab.cos(theta/2)])]
        )


class RotationZ(OneQubitRotationGate):
    '''A one qubit rotation along the Pauli-Z axis.'''
    _label = 'RZ'

    def _generator(self, theta, slices=None):
        return ab.vstack(
            [ab.hstack([ab.exp(-1j*theta/2), 0]),
             ab.hstack([0, ab.exp(1j*theta/2)])]
        )


class ControlledOneQubitGate(Gate):
    '''Abstract base class for one qubit gate with one control qubit. When
    placing them in a circuit, the first qubit is the control qubit, the second
    qubit is the target.'''

    def to_tsrex(self, qubits):
        if qubits[0] < qubits[1]:
            return (
                    _proj0 &
                    eye(2**(qubits[1] - qubits[0]))
                   ) + (
                    _proj1 &
                    eye(2**(qubits[1] - qubits[0] - 1)) &
                    self._tsrex
                   )
        return (
                eye(2**(qubits[0] - qubits[1])) &
                _proj0
               ) + (
                self._tsrex &
                eye(2**(qubits[0] - qubits[1] - 1)) &
                _proj1
               )


class CX(ControlledOneQubitGate):
    '''A controlled-X (CNOT) gate.'''
    def __init__(self):
        self._tsrex = PauliX().to_tsrex()
        self._label = 'CX'
