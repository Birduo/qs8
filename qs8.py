# qs8: Birduo's Quantum Statevector Simulator

import numpy as np
import random
from icecream import ic

# basic gate init for easy access
H = np.asarray([[1, 1], [1, -1]]) * 2**-.5
X = np.asarray([[0, 1], [1, 0]])
Y = np.asarray([[0, -1j], [1j, 0]])
Z = np.asarray([[1, 0], [0, -1]])

# this has the first qb listed as control
CX = np.asarray([[1, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0]])

# this one has second qb control
CXl = np.asarray([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

SWP = np.asarray([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])

# how 2 store gate:
# dict w/ (gate: np.ndarray, qubits: list[int])
# current qubits acting on the gate

class QCirc:
    def __init__(self, qubits: int):
        self.qubits = qubits
        # state will be array stored in [2*qb#:2*(qb#+1)] format
        # that will splice the correct part of the statevector
        self._state = np.append(np.ones(1), np.zeros(((2**qubits)-1, 1)))
        self._circuit = []
        self._circ_mat = np.eye(2**qubits, dtype=complex) # circuit matrix

    # adds the gate to the proper column in the gates list
    def set_gate(self, gate: np.ndarray, qubits: list[int], column: int):
        if gate.shape[0] == 2**len(qubits):
            if len(qubits) == 0:
                ic("Invalid qubit gate (Can't be empty list!)")
                return None
            elif len(qubits) <= 0 or len(qubits) > self.qubits:
                print("Qubit length out of range!")
                return None

            # users inferred to SWAP themselves
            # therefore, all qubits should be next to each other
            dists = [j-i for i, j in zip(qubits[:-1], qubits[1:])] 
            if not (dists == [1] * (len(qubits) - 1)):
                print("Invalid distances!")
                return None

            # adding necessary columns if needed by column index
            while len(self._circuit) <= column:
                self._circuit.append([])

            # if the qubit is already operated on, NOP
            for gate in self._circuit[column]:
                for qb in gate[1]:
                    if qb in qubits:
                        print(f"Duplicate qubit operated on q{qb}!")

                        return None

            self._circuit[column].append((gate, qubits))
            # sort by qubit order
            self._circuit[column].sort(key=lambda tup: tup[1][0])

        else:
            print(f"Gate shape {gate.shape} does not match 2^qbs supplied!")

    # interprets column into a single matrix
    def _col_to_mat(self, column: int):
        col_mat = []
        qb = 0

        if column == 0 and len(self._circuit) == 0:
            return np.eye(2**self.qubits, dtype=complex)

        # getting initial state of col_mat
        if len(self._circuit[column]) > 0:
            if qb in self._circuit[column][0][1]:
                col_mat = self._circuit[column][0][0]
                # advance qb by length of the qubits selected by gate
                qb += len(self._circuit[column][0][1])
            else:
                col_mat = np.eye(2, dtype=complex)
                qb += 1
        elif len(self._circuit[column]) == 0:
            return np.eye(2**self.qubits, dtype=complex)
 
        while qb < self.qubits:
            for gate, qubits in self._circuit[column]:
                if qb in qubits:
                    col_mat = np.kron(gate, col_mat)
                    qb += len(qubits)
                else:
                    col_mat = np.kron(np.eye(2, dtype=complex), col_mat)
                    qb += 1

        return col_mat

    def init_state(self, state=None):
        if state is None:
            self._state = np.append(np.ones(1), np.zeros(((2**self.qubits)-1, 1)))
        else:
            if state.shape[0] == 2**self.qubits:
                self._state = state
            else:
                return None

    # evolve state by given column
    def _interpret_column(self, column: int):
        self._state = self._col_to_mat(column) @ self._state

    # public-facing portion of interpreting
    def interpret_circuit(self):
        self.init_state()

        for col in range(len(self._circuit)):
            self._interpret_column(col)

    # potential for renaming to transpile
    def build_circuit_matrix(self):
        if self.get_columns() < 1:
            return ValueError
        
        for col in range(self.get_columns()):
            mat = self._col_to_mat(col)
            if col == 0:
                self._circ_mat = mat
            else:
                try:
                    self._circ_mat = self._circ_mat @ mat
                except:
                    raise ValueError("Something changed the col to mat type!")

    ## basic helper functions below ##
    def get_circuit_matrix(self):
        return self._circ_mat

    def get_columns(self) -> int:
        return len(self._circuit)

    def get_pr(self):
        return np.real(self._state**2)

    def get_sv(self):
        return self._state
    
    # similar to Vinny's implementation
    # choose a random number and see if it lands
    # between the given probabilities
    def measure_pr(self):
        choice = random.random()

        pr = self.get_pr()
        pr_cumsum = np.cumsum(pr)
        
        prev_pr = 0
        for index, pr in enumerate(pr_cumsum):
            if prev_pr <= choice <= pr:
                return index

            prev_pr = pr
        
        return -1

    # runs the circuit by the built circuit matrix
    # initializes state as well !
    # if shots are defined, then return counts
    # output: (state, counts)
    def run_circuit(self, state=None, build=False):
        if state is None:
            self.init_state()
        else:
            self.init_state(state)

        if build:
            self._circ_mat = self.build_circuit_matrix()

        # evaluate state by circuit matrix
        try:
            self._state = self._state @ self._circ_mat
        except:
            raise ValueError

    def get_counts(self, shots, dictionary=True):
        if dictionary:
            counts = {}
        else:
            counts = np.zeros((2**self.qubits))

        for _ in range(shots):
            out = self.measure_pr()
            if dictionary:
                if str(out) in counts:
                    counts[str(out)] += 1
                else:
                    counts[str(out)] = 1

            else: counts[out] += 1

        return counts
    
    def __str__(self) -> str:
        circ = [""] * self.qubits * 2

        # adding initial qubit labels
        for qb in range(self.qubits * 2):
            if qb % 2 == 0:
                circ[qb] += f"q{qb//2} "
            else:
                circ[qb] += " " * len(f"q{qb//2} ")
        
        # add to circuit column by column
        for col in self._circuit:
            # adding gates
            for qb in range(self.qubits):
                for gate, qubits in col:
                    # gate detection
                    if qb == qubits[0]:
                        if gate.shape == H.shape and (gate == H).all():
                            circ[qb * 2] += "H-"
                        elif gate.shape == X.shape and (gate == X).all():
                            circ[qb * 2] += "X-"
                        elif gate.shape == Y.shape and (gate == Y).all():
                            circ[qb * 2] += "Y-"
                        elif gate.shape == Z.shape and (gate == Z).all():
                            circ[qb * 2] += "Z-"
                        elif gate.shape == CX.shape and (gate == CX).all():
                            circ[qb * 2] += "CX"
                        elif gate.shape == SWP.shape and (gate == SWP).all():
                            circ[qb * 2] += "SW"
                        else:
                            circ[qb * 2] += "U3"

                    # continue the gate
                    elif qb in qubits:
                        circ[qb * 2 - 1] += "| "
                        circ[qb * 2] += "o-"

            max_len = max([len(row) for row in circ])
            for qb in range(self.qubits * 2):
                if len(circ[qb]) < max_len:
                    diff = max_len - len(circ[qb])
                    circ[qb] += ("-" if qb % 2 == 0 else " ") * diff

            # add everything to the rest of the length here for this col
            # get the max length of circ and even everything out

        out = "\n".join(circ)
        
        return out

# actual test code
def main():
    qb = 4
    qc = QCirc(qb)

    # auto-ghz up to 13 qb

    qc.set_gate(X, [0], 0)
    qc.set_gate(H, [0], 1)

    for n in range(qb-1):
        qc.set_gate(CX, [n, n + 1], qc.get_columns() + 1)

    qc.build_circuit_matrix()

    qc.run_circuit()

    counts = qc.get_counts(1000)
    print(counts)

    print(str(qc))

if __name__ == "__main__":
    main()
