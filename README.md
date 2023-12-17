# qs8 Quantum Statevector Simulator
Using numpy to create a statevector-based simulator

### How does it work?
Circuits are organized into different columns, where you can place each gate in a column on an array of qubits.
You can then interpret or build/run the circuit:
- Interpreting will evolve the state column by column.
- Building (and running) will build a matrix of the entire circuit and then advance the statevector by that matrix.

### How do I use it?
See example.py for this example!
```python
from qs8 import *
qb = 4
# creating a circuit w/ 4 qubits
qc = QCirc(qb)

# setting the gate on q0 in column 0 to X
qc.set_gate(X, [0], 0)
# setting the gate on q0 in column 1 to H
qc.set_gate(H, [0], 1)

# create CX gates linking from q0 to qb-1 (q3)
for i in range(qb - 1):
    qc.set_gate(CX, [i, i + 1], i + 2)

# builds and runs circuit
qc.run_circuit(build=True)
# interprets circuit column by column
# qc.interpret_circuit()

# getting a dictionary of counts
counts = qc.get_counts(10000)

# print the circuit and counts
print(str(qc))
print(counts)
```

### Python Dependencies
- numpy

### Roadmap
- [x] Adding standard or custom gates to a circuit
- [x] Interpreting single gates
- [x] Interpreting local multi-qubit gates
- [x] Store circuit as unitary operation
- [x] Initialize functionality
- [ ] Advanced state preparation
- [ ] Non-local multi-qubit gates
- [ ] Customizable dtype (complex128 is currently used)

#### Why'd you name it qs8?
I used an 8 in a username last year and it also kinda looks like a sideways infinity.