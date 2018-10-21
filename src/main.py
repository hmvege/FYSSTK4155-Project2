from lib import ising_1d as ising
import numpy as np


def task1a():
    N_samples = 100

    np.random.seed(12)

    # system size
    L = 40

    # create 10000 random Ising states
    states = np.random.choice([-1, 1], size=(10000, L))

    # calculate Ising energies
    energies = ising.ising_energies(states, L)



    print(energies.shape)


def main():
    task1a()



if __name__ == '__main__':
    main()