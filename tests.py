from mps_gradient import *

def basic_test():
    for num_qubits in range(4,10):
        print(f'\rTesting Num Qubits: {num_qubits}',end='')
        # test that identity block + inverted block gives the identity
        qs = cirq.GridQubit.rect(1,num_qubits)

        num_blocks_layer_1 = len(qs)    // 2
        num_blocks_layer_2 = len(qs)-1  // 2

        p = np.random.rand((8 * num_blocks_layer_1) + (4*num_blocks_layer_2))
        circuit = cirq.Circuit()

        circuit.append(random_block_layer(qs, p), strategy=InsertStrategy.EARLIEST)
        circuit.append(inverted_block(qs, p), strategy=InsertStrategy.EARLIEST)
        assert np.allclose(cirq.unitary(circuit), np.eye(2**num_qubits))

        # mps tests: check that the circuit mps gives the same exp values as classical mps
        qs = cirq.GridQubit.rect(1,num_qubits)
        circuit = cirq.Circuit()

        # create the classical MPS
        mps = fMPS().random(L=num_qubits, d=2, D=2)
        mps.left_canonicalise()

        # build a circuit with the MPS
        circuit.append(mps_block(qs, mps))

        # simulat the circuit to find the state vector
        sim = cirq.Simulator()
        psi = sim.simulate(circuit).final_state_vector
        H = np.random.rand(2**num_qubits, 2**num_qubits)
        H = H + H.conj().T

        # check the simulated expectation value is the same as the mps one
        assert np.allclose(np.abs(psi @ H @ psi.conj().T), np.abs(mps.E_L(H)))

        # check that with identity blocks the above tests still hold
        mps = fMPS().random(L=num_qubits, d=2, D=2)
        mps.left_canonicalise()
        
        # build circuit with identity block, mps, and iden block
        circuit = cirq.Circuit()
        circuit.append(identity_block(qs))
        circuit.append(mps_block(qs, mps))
        circuit.append(identity_block(qs))

        sim = cirq.Simulator()
        psi = sim.simulate(circuit).final_state_vector
        H = np.random.rand(2**num_qubits, 2**num_qubits)
        H = H + H.conj().T

        assert np.allclose(np.abs(psi @ H @ psi.conj().T), np.abs(mps.E_L(H)))
