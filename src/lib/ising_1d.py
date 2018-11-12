import numpy as np
import scipy.sparse as sp


def ising_energies(states, L):
    """
    This function calculates the energies of the states in the nn Ising 
    Hamiltonian.
    """
    J = np.zeros((L, L),)

    for i in range(L):
        J[i, (i+1) % L] -= 1.0

    # compute energies
    E = np.einsum('...i, ij, ...j->...', states, J, states)

    return E


def generate_1d_ising_data(L_system_size, N_samples):
    np.random.seed(1234)

    # system size
    L = L_system_size

    # create 10000 random Ising states
    states = np.random.choice([-1, 1], size=(N_samples, L))

    # calculate Ising energies
    energies = ising_energies(states, L)
    energies = energies.reshape((energies.shape[0], 1))

    # reshape Ising states into RL samples: S_iS_j --> X_p
    states = np.einsum('...i,...j->...ij', states, states)

    # Reshaping to correspond to energies.
    # Shamelessly stolen a lot of from:
    # https://physics.bu.edu/~pankajm/ML-Notebooks/HTML/NB_CVI-linreg_ising.html
    # E.g. why did no-one ever tell me about einsum?
    # That's awesome - no way I would have discovered that by myself.
    stat_shape = states.shape
    states = states.reshape((stat_shape[0], stat_shape[1]*stat_shape[2]))

    return states, energies


def main():
    # Basic test case for 1d ising model parameters

    # define Ising model aprams

    np.random.seed(12)

    # system size
    L = 40

    # create 10000 random Ising states
    states = np.random.choice([-1, 1], size=(10000, L))

    # calculate Ising energies
    energies = ising_energies(states, L)
    print(energies)



if __name__ == '__main__':
    main()
