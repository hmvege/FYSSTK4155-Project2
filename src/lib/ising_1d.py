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
