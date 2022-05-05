import matplotlib.pyplot as plt
from tqdm import tqdm
import cirq
import numpy as np
from mps_gradient import (
    energy, 
    energy_circuit,
    get_opt_mps,
    generate_params,
    full_ham,
    energy_grad_circuit_f_diff,
    energy_grad_f_diff
)
from xmps.fMPS import fMPS
from xmps.Hamiltonians import Hamiltonian
from xmps.fTDVP import Trajectory
from copy import deepcopy


def test_a_bunch_of_stuff():
    # define tfim Hamiltonian and chain length
    tfim = Hamiltonian({'ZZ':1,'X':1,'Z':0.5})
    L = 3

    # build the identity circuit
    # to do this we define the 'forward' params which define the odd layers of XPow and CZ gates
    # If we want the identity we copy these to the backward params
    # Otherwise we define new parameters

    # generate random params
    params_forward = generate_params(L)

    # use the same params to cancel out
    params_backward = deepcopy(params_forward)

    identity_energy_circuit = energy_circuit(
        mps = None,
        L = L,
        p_forward = params_forward,
        p_reverse = params_backward,
        use_mps = False
    )

    # check this is the identity
    assert np.allclose(np.eye(2**L), cirq.unitary(identity_energy_circuit))

    iden_energy = energy(tfim, identity_energy_circuit)

    print('Identity Energy:')
    print(iden_energy)

    # this is the identity energy I think
    assert np.allclose( iden_energy , 1.1666646)

    # Now generate new backwards params
    random_params_backward = generate_params(L)

    random_energy_circuit = energy_circuit(
        mps = None,
        L = L,
        p_forward = params_forward,
        p_reverse = random_params_backward,
        use_mps = False
    )

    print('\nRandom Circuit Energy:')
    print(energy(tfim, random_energy_circuit))

    # insert a random MPS now into the circuit and check this works as expected

    random_mps = fMPS().random(L=L, d=2, D=2).left_canonicalise()

    random_mps_energy = random_mps.E_L(full_ham(tfim, L))

    print('\nRandom MPS Energy:')
    print(random_mps_energy)

    random_mps_circuit = energy_circuit(
        mps = random_mps,
        L = L,
        p_forward = params_forward,
        p_reverse = params_backward,
        use_mps = True
    )

    random_mps_circuit_energy = energy(tfim, random_mps_circuit)

    print('\nRandom MPS Circuit Energy:')
    print(random_mps_circuit_energy)

    assert np.allclose(random_mps_circuit_energy, random_mps_energy)

    print('\nFind Optimized MPS:')
    # now optimize an mps using fTDVP and 
    optimized_mps, _ = get_opt_mps(tfim, L, T = 0.5, steps = 50)

    # check optimized energy
    optimized_mps_energy = optimized_mps.E_L(full_ham(tfim, L))

    print('\nOptimized MPS Energy:')
    print(optimized_mps_energy)

    # build this into a circuit

    optimized_mps_circuit = energy_circuit(
        mps = optimized_mps,
        L = L,
        p_forward = params_forward,
        p_reverse = params_backward,
        use_mps = True
    )

    # check the energy of the optimized circuit
    optimized_mps_circuit_energy = energy(tfim, optimized_mps_circuit)

    print('\nOptimized MPS Circuit Energy:')
    print(optimized_mps_circuit_energy)

    assert np.allclose(optimized_mps_circuit_energy, optimized_mps_energy)

    # check the gradients of these circuits:
    # Make a plus and minus circuit to get the differences
    
    eps = 1e-5

    identity_p, identity_m = energy_grad_circuit_f_diff(
        mps = None,
        L = L,
        p_forward = params_forward,
        p_reverse = params_backward,
        block_num = 1, param_index = 5,
        use_mps = False,
        eps = eps 
    )

    assert np.allclose(np.eye(2**L), cirq.unitary(identity_m))

    assert (not np.allclose(np.eye(2**L), cirq.unitary(identity_p)))

    grad = energy_grad_f_diff(tfim, identity_p, identity_m, eps)

    print('\nIdentity Gradient:')
    print(grad)

    # random_circuit:

    random_p, random_m = energy_grad_circuit_f_diff(
        mps = None,
        L = L,
        p_forward = params_forward,
        p_reverse = random_params_backward,
        block_num = 1, param_index = 5,
        use_mps = False,
        eps = eps 
    )

    grad = energy_grad_f_diff(tfim, random_p, random_m, eps)

    print('\nRandom Gradient:')
    print(grad)

    # optimized_mps:

    opt_p, opt_m = energy_grad_circuit_f_diff(
        mps = optimized_mps,
        L = L,
        p_forward = params_forward,
        p_reverse = params_backward,
        block_num = 1, param_index = 5,
        use_mps = True,
        eps = eps 
    )

    grad = energy_grad_f_diff(tfim, opt_p, opt_m, eps)

    print('\nOptimized Gradient:')
    print(grad)


def identity_variance_data(ham, L, block_num, param_index, eps = 1e-5, reps = 100):

    res = []

    for i in tqdm(range(reps)):
        
        # generate random params
        p_forward = generate_params(L)

        # use the same params to cancel out
        p_reverse = deepcopy(p_forward)

        circuit_p, circuit_m = energy_grad_circuit_f_diff(
            mps = None,
            L = L,
            p_forward = p_forward,
            p_reverse = p_reverse,
            block_num = block_num, param_index = param_index,
            use_mps = False,
            eps = eps
        )

        res.append(energy_grad_f_diff(ham, circuit_p, circuit_m, eps))

    return res


def random_variance_data(ham, L, block_num, param_index, eps = 1e-5, reps = 100):

    res = []

    for i in tqdm(range(reps)):
        
        # generate random params
        p_forward = generate_params(L)

        # use the same params to cancel out
        p_reverse = generate_params(L)

        circuit_p, circuit_m = energy_grad_circuit_f_diff(
            mps = None,
            L = L,
            p_forward = p_forward,
            p_reverse = p_reverse,
            block_num = block_num, param_index = param_index,
            use_mps = False,
            eps = eps
        )

        res.append(energy_grad_f_diff(ham, circuit_p, circuit_m, eps))

    return res


def optimized_variance_data(mps, ham, L, block_num, param_index, eps=1e-5, reps = 100):
    res = []

    for i in tqdm(range(reps)):
        
        # generate random params
        p_forward = generate_params(L)

        # use the same params to cancel out
        p_reverse = deepcopy(p_forward)

        circuit_p, circuit_m = energy_grad_circuit_f_diff(
            mps = mps,
            L = L,
            p_forward = p_forward,
            p_reverse = p_reverse,
            block_num = block_num, param_index = param_index,
            use_mps = True,
            eps = eps
        )

        res.append(energy_grad_f_diff(ham, circuit_p, circuit_m, eps))

    return res


def variance_exp():
    # define tfim Hamiltonian and chain length
    tfim = Hamiltonian({'ZZ':1,'X':1,'Z':0.5})
    L = 3

    # build the identity circuit
    # to do this we define the 'forward' params which define the odd layers of XPow and CZ gates
    # If we want the identity we copy these to the backward params
    # Otherwise we define new parameters

    mps, _ = get_opt_mps(tfim, L, T=0.4, steps = 40)

    results = []
    Ls = [4,6,8,10,12]
    for L in Ls:
        print(f'L = {L}')
        # mps, _ = get_opt_mps(tfim, L, T=0.4, steps = 40)

        data = random_variance_data(
            ham = tfim, 
            L = L,
            block_num = 1, param_index = 5,
        )

        var = np.var(data)
        results.append(var)

    np.save('./randomized_var.npy', results)
    plt.plot(Ls, results)
    plt.show()


if __name__ == '__main__':

    Ls = [4,6,8,10,12]
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    # for Palatino and other serif fonts use:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
    })
    # It's also possible to use the reduced notation by directly setting font.family:
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
    })

    ran = np.load('./randomized_var.npy')
    opt = np.load('./optimized_var.npy')

    plt.plot(Ls, ran, label = 'Random')
    plt.plot(Ls, opt, label = 'MPS')
    plt.ylabel(r'$Var( \frac{dE}{d\theta} )$', fontsize=20)
    plt.xlabel('Qubit Number', fontsize=20)

    plt.yscale('log')
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()



