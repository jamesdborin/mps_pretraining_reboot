import cirq
import numpy as np
from cirq.circuits import InsertStrategy
from scipy.linalg import null_space
from xmps.fMPS import fMPS
from functools import reduce
from copy import deepcopy
from xmps.fTDVP import Trajectory
import qsimcirq

def random_block_layer(qs, p):
    """Add rx-rx-cz-rx-rx two qubit gates in a brick wall fashion."""
    num_blocks_layer_1 = len(qs)    // 2
    num_blocks_layer_2 = (len(qs)-1)  // 2
    
    for i in range(num_blocks_layer_1):
        p_block = i*4
        yield [
            cirq.XPowGate(exponent = p[p_block + 0]).on(qs[2*i + 0]),
            cirq.XPowGate(exponent = p[p_block + 1]).on(qs[2*i + 1]),
            cirq.CZ(qs[2*i + 0], qs[2*i + 1]),
            cirq.XPowGate(exponent = p[p_block + 2]).on(qs[2*i + 0]),
            cirq.XPowGate(exponent = p[p_block + 3]).on(qs[2*i + 1]),
        ]

    p_new = 4 * num_blocks_layer_1

    for i in range(num_blocks_layer_2):
        p_block = i*4
        yield [
            cirq.XPowGate(exponent = p[p_new + p_block + 0]).on(qs[2*i + 0 + 1]),
            cirq.XPowGate(exponent = p[p_new + p_block + 1]).on(qs[2*i + 1 + 1]),
            cirq.CZ(qs[2*i + 0 + 1], qs[2*i + 1 + 1]),
            cirq.XPowGate(exponent = p[p_new + p_block + 2]).on(qs[2*i + 0 + 1]),
            cirq.XPowGate(exponent = p[p_new + p_block + 3]).on(qs[2*i + 1 + 1]),
        ]

    p_new = (4 * num_blocks_layer_1) + (4 * num_blocks_layer_2)

    for i in range(num_blocks_layer_1):
        p_block = i*4
        
        yield [
            cirq.XPowGate(exponent = p[p_new + p_block + 0]).on(qs[2*i + 0]),
            cirq.XPowGate(exponent = p[p_new + p_block + 1]).on(qs[2*i + 1]),
            cirq.CZ(qs[2*i + 0], qs[2*i + 1]),
            cirq.XPowGate(exponent = p[p_new + p_block + 2]).on(qs[2*i + 0]),
            cirq.XPowGate(exponent = p[p_new + p_block + 3]).on(qs[2*i + 1]),
        ]


def inverted_block(qs, p):
    """Add rx-rx-cz-rx-rx two qubit gates in a brick wall fashion."""
    num_blocks_layer_1 = len(qs)    // 2
    num_blocks_layer_2 = (len(qs)-1)  // 2

    p_new = (4 * num_blocks_layer_1) + (4 * num_blocks_layer_2)

    for i in range(num_blocks_layer_1):
        p_block = i*4
        yield [
            cirq.XPowGate(exponent = -p[p_new + p_block + 2]).on(qs[2*i + 0]),
            cirq.XPowGate(exponent = -p[p_new + p_block + 3]).on(qs[2*i + 1]),
            cirq.CZ(qs[2*i + 0], qs[2*i + 1]),
            cirq.XPowGate(exponent = -p[p_new + p_block + 0]).on(qs[2*i + 0]),
            cirq.XPowGate(exponent = -p[p_new + p_block + 1]).on(qs[2*i + 1]),
        ]

    p_new = 4 * num_blocks_layer_1

    for i in range(num_blocks_layer_2):
        p_block = i*4
        yield [
            cirq.XPowGate(exponent = -p[p_new + p_block + 2]).on(qs[2*i + 0 + 1]),
            cirq.XPowGate(exponent = -p[p_new + p_block + 3]).on(qs[2*i + 1 + 1]),
            cirq.CZ(qs[2*i + 0 + 1], qs[2*i + 1 + 1]),
            cirq.XPowGate(exponent = -p[p_new + p_block + 0]).on(qs[2*i + 0 + 1]),
            cirq.XPowGate(exponent = -p[p_new + p_block + 1]).on(qs[2*i + 1 + 1]),
        ]


    for i in range(num_blocks_layer_1):
        p_block = i*4
        
        yield [
            cirq.XPowGate(exponent = -p[p_block + 2]).on(qs[2*i + 0]),
            cirq.XPowGate(exponent = -p[p_block + 3]).on(qs[2*i + 1]),
            cirq.CZ(qs[2*i + 0], qs[2*i + 1]),
            cirq.XPowGate(exponent = -p[p_block + 0]).on(qs[2*i + 0]),
            cirq.XPowGate(exponent = -p[p_block + 1]).on(qs[2*i + 1]),
        ]


def identity_block(qs):
    
    num_blocks_layer_1 = len(qs)    // 2
    num_blocks_layer_2 = len(qs)-1  // 2

    p = np.random.rand((8 * num_blocks_layer_1) + (4*num_blocks_layer_2))

    yield random_block_layer(qs, p)
    yield inverted_block(qs, p)


def tensor_to_unitary(t):
    D,L,R = t.shape
    
    A = t.transpose([1,0,2]).reshape(D*L,R)
    uNull = null_space(A.conj().T)
    
    return np.concatenate([A, uNull], axis=1)


def Compile(u):
    """Compile a 2-qubit unitary into a 15 parameter vector to feed into the KAK decomposition"""
    q = int(np.log2(u.shape[0]))

    if q == 1:
        return cirq.linalg.deconstruct_single_qubit_matrix_into_angles(u)

    if q == 2:
        kak = cirq.linalg.kak_decomposition(u)
        b0,b1 = kak.single_qubit_operations_before
        a0,a1 = kak.single_qubit_operations_after
        x,y,z = kak.interaction_coefficients

        b0p = cirq.linalg.deconstruct_single_qubit_matrix_into_angles(b0)
        b1p = cirq.linalg.deconstruct_single_qubit_matrix_into_angles(b1)
        a0p = cirq.linalg.deconstruct_single_qubit_matrix_into_angles(a0)
        a1p = cirq.linalg.deconstruct_single_qubit_matrix_into_angles(a1)

        return [*b0p] + [*b1p] + [x,y,z] + [*a0p] +[*a1p]


class KakDecomp(cirq.Gate):
    def __init__(self, params):
        self.params = params
        self.q = 2
        self.s = ["MPS"] 
        
    def num_qubits(self):
        return self.q
    
    def _circuit_diagram_info_(self, args):
        return self.s*self.q
    
    def _decompose_(self, qubits):
        if self.q == 2:
            # b0 decomp
            yield cirq.rz(self.params[0]).on(qubits[0])
            yield cirq.ry(self.params[1]).on(qubits[0])
            yield cirq.rz(self.params[2]).on(qubits[0])

            # b1 decomp
            yield cirq.rz(self.params[3]).on(qubits[1])
            yield cirq.ry(self.params[4]).on(qubits[1])
            yield cirq.rz(self.params[5]).on(qubits[1])

            # interaction gates
            yield cirq.XXPowGate(exponent=-2*self.params[6]/np.pi).on(*qubits)
            yield cirq.YYPowGate(exponent=-2*self.params[7]/np.pi).on(*qubits)
            yield cirq.ZZPowGate(exponent=-2*self.params[8]/np.pi).on(*qubits)
            
            # a0 decomp
            yield cirq.rz(self.params[9]).on(qubits[0])
            yield cirq.ry(self.params[10]).on(qubits[0])
            yield cirq.rz(self.params[11]).on(qubits[0])

            # a1 decomp
            yield cirq.rz(self.params[12]).on(qubits[1])
            yield cirq.ry(self.params[13]).on(qubits[1])
            yield cirq.rz(self.params[14]).on(qubits[1])


class SingleMPS(cirq.Gate):
    def __init__(self, params):
        self.p = params
        
    def num_qubits(self):
        return 1
    
    def _circuit_diagram_info_(self, args):
        return ["MPS"]
    
    def _decompose_(self, qubits):
        yield cirq.rz(self.p[0]).on(qubits[0])
        yield cirq.ry(self.p[1]).on(qubits[0])
        yield cirq.rz(self.p[2]).on(qubits[0])


def mps_block(qs, mps):
    """apply diagonal kak decomposition gates defined by mps_params."""
    
    num_2_q_tensors = len(qs)-1
    
    for i in range(num_2_q_tensors):
        mps_tensor = tensor_to_unitary(mps.data[-1-i])
        mps_params = Compile(mps_tensor)

        qubit_num = num_2_q_tensors - i - 1
        
        yield KakDecomp(mps_params).on(qs[qubit_num], qs[qubit_num+1])

    mps_tensor = tensor_to_unitary(mps.data[0])
    mps_params = Compile(mps_tensor)

    yield SingleMPS(mps_params[-3:]).on(qs[0])
    

def full_ham(ham, L):
    matrices = ham.to_matrices(L)
    full = np.zeros([2**L, 2**L]).astype(np.complex64)

    I = np.eye(2)
    for i,m in enumerate(matrices):
        
        full += reduce(np.kron, [I]*i + [m] + [I]*(L-i-2)  )

    return full / L


def get_opt_mps(ham, L, T = 10, steps = 500):
    """Get an optimized mps of length L for a given hamiltonian"""
    mps = fMPS().random(L=L, d=2, D=2).left_canonicalise()
    H = ham.to_matrices(L)
    traj = Trajectory(mps, H=H).eulerint(-1j*np.linspace(0,T,steps))
    mps_opt = traj.mps

    return mps_opt, traj


def energy_circuit(mps, L, p_forward, p_reverse, use_mps = True):
    qs = cirq.GridQubit.rect(1,L)
    circuit = cirq.Circuit()

    first_depth = int(np.ceil(L/2))
    second_depth = int(np.floor(L/2))
    
    assert len(p_forward) == first_depth + second_depth
    assert len(p_reverse) == first_depth + second_depth

    for i in range(first_depth):
        circuit.append(random_block_layer(qs,p_forward[i]))
        circuit.append(inverted_block(qs,p_reverse[i]))

    if use_mps:
        circuit.append(mps_block(qs, mps))

    for i in range(second_depth):
        circuit.append(random_block_layer(qs,p_forward[first_depth + i - 1]))
        circuit.append(inverted_block(qs,p_reverse[first_depth + i - 1]))

    return circuit


def energy(ham, circuit):
    sim = qsimcirq.QSimSimulator()
    res = sim.simulate(circuit).final_state_vector
    
    L = len(circuit.all_qubits())
    H = full_ham(ham, L)

    E = res @ H @ res.conj().T

    return E.real


def generate_params(L):
    num_blocks_layer_1 =  L     // 2
    num_blocks_layer_2 = (L-1)  // 2

    p = [np.random.rand((8 * num_blocks_layer_1) + (4*num_blocks_layer_2)) for _ in range(L)]

    return p


def energy_grad_circuit_param_shift(mps, L, p_forward, p_reverse, block_num, param_index, use_mps = True):

    p_f_plus = deepcopy(p_forward)
    p_f_minus = deepcopy(p_forward)

    p_f_plus[block_num][param_index] += np.pi/2
    p_f_minus[block_num][param_index] -= np.pi/2

    plus_circuit = energy_circuit(mps, L, p_f_plus, p_reverse, use_mps=use_mps)
    minus_circuit = energy_circuit(mps, L, p_f_minus, p_reverse, use_mps=use_mps)

    return plus_circuit, minus_circuit


def energy_grad_circuit_f_diff(mps, L, p_forward, p_reverse, block_num, param_index, eps = 1.5e-8, use_mps = True):
    """Use forward difference to estimate the gradient"""
    
    p_f_plus = deepcopy(p_forward)
    p_f_plus[block_num][param_index] += eps

    plus_circuit = energy_circuit(mps, L, p_f_plus, p_reverse, use_mps=use_mps)
    minus_circuit = energy_circuit(mps, L, p_forward, p_reverse, use_mps=use_mps)

    return plus_circuit, minus_circuit


def energy_grad_param_shift(ham, plus_circuit, minus_circuit):

    grad = (1/2) * (
        energy(ham, plus_circuit) - \
        energy(ham, minus_circuit)
        )

    return grad


def energy_grad_f_diff(ham, plus_circuit, minus_circuit, eps = 1.5e-8):

    grad = (1/eps) * (
        energy(ham, plus_circuit) - \
        energy(ham, minus_circuit)
        )

    return grad



def calc_variance_of_grad(L, ham, block_num, param_index, use_mps = True, opt_mps = True, identity = True):
    
    if opt_mps and use_mps:
        mps, _ = get_opt_mps(ham, L, T=0.4, steps = 80)

    else:
        mps = fMPS().random(L=L, d=2, D=2).left_canonicalise()

    grads = [energy_grad(mps, ham, L, block_num, param_index, use_mps=use_mps, identity=identity) for _ in range(200)]

    grad_var = np.var(grads)

    return grad_var

