#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numbers import Integral
from funfact.lang._special import eye
from funfact.util.integral import check_bounded_integral


class Circuit():
    '''A class to create FunFact quantum circuits.'''
    def __init__(self, nbqubits: Integral, offset=0):
        '''
        Args:
            nbqubits: int
                The number of qubits (or size) of this quantum circuit. The
                qubit indices will range by default from 0 to nbqubits-1.
            offset: int
                The offset on the qubits of this circuit. This optional
                argument modifies the range of qubits of this circuit to
                [0, self._nbqubits-1] + self._offset
        '''
        check_bounded_integral(nbqubits, minv=1)
        check_bounded_integral(offset, minv=0)
        self._nbqubits = nbqubits
        self._offset = offset
        self._qgates = []

    @property
    def nbqubits(self):
        '''The number of qubits (size) of this quantum circuit.'''
        return self._nbqubits

    @property
    def qubits(self):
        return [self._offset, self._nbqubits - 1 + self._offset]

    def append(self, qobject):
        '''Add a quantum object at the end of this circuit.'''
        self._qgates.append(qobject)

    def to_tsrex(self):
        '''Generate a tensor expression for the circuit.

        Returns:
            tsrex:
                tensor expression for this circuit
        '''
        tsrex = eye(2**self._nbqubits)
        for i, g in enumerate(self._qgates):
            if g.qubits[-1] >= self.nbqubits:
                raise RuntimeError(
                    f"Gate {i} acts on qubit {g.qubits[-1]}, which is out of "
                    f"bounds for circuit with {self.nbqubits} qubits."
                )
            tsrex = (
                eye(2**g.qubits[0]) &
                g.to_tsrex() &
                eye(2**(self.nbqubits - g.qubits[-1] - 1))
            ) @ tsrex
        return tsrex
