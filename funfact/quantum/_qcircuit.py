#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Integral
from funfact.lang._special import eye
from ._qgates import _PlacedGate


class Circuit():
    '''A class to create FunFact quantum circuits.'''
    def __init__(self, nbqubits: Integral):
        '''
        Args:
            nbqubits: int
                The number of qubits (or size) of this quantum circuit. The
                qubit indices will range by default from 0 to nbqubits-1.
        '''
        self._nbqubits = nbqubits
        self._qgates = []

    @property
    def nbqubits(self):
        '''The number of qubits (size) of this quantum circuit.'''
        return self._nbqubits

    @property
    def qubits(self):
        return [0, self._nbqubits - 1]

    def append(self, qobject, at=None):
        '''Add a quantum object at the end of this circuit.'''
        if isinstance(qobject, _PlacedGate):
            self._qgates.append(qobject)
        elif at:
            self._qgates.append(qobject @ at)

    def to_tsrex(self):
        '''Generate a tensor expression for the circuit.

        Returns:
            tsrex:
                tensor expression for this circuit
        '''
        tsrex = eye(2**self._nbqubits)
        for i, g in enumerate(self._qgates):
            qubits = sorted(g.qubits)
            if qubits[-1] >= self.nbqubits:
                raise RuntimeError(
                    f"Gate {i} acts on qubit {qubits[-1]}, which is out of "
                    f"bounds for circuit with {self.nbqubits} qubits."
                )
            tsrex = (
                eye(2**qubits[0]) &
                g.to_tsrex() &
                eye(2**(self.nbqubits - qubits[-1] - 1))
            ) @ tsrex
        return tsrex
