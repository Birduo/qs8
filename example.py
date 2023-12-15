from qs8 import *

# example code to get started with!
def main():
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

if __name__ == "__main__":
    main()
