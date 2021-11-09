from random_walk import random_walk
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def random_walk_100_steps():
    """
    PART 1: Plot a random walk of 100 steps
    """

    # Make the results reproducible by specifying the seed
    np.random.seed(1)

    x_locations, y_locations, r = random_walk(100, True)

    fig, ax = plt.subplots()
    ax.plot([0], [0], 'k.', label='Start')
    ax.plot(x_locations, y_locations, '-b', label='Path')
    ax.plot(x_locations[-1], y_locations[-1], 'r.', label='End')
    ax.set_title(f"Random walk of n = 100 steps, (x0, y0) = (0,0). r = {round(r, 2)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc='best')
    fig.show()
    # fig.savefig("./figs/1.png", dpi=300)


def find_rms_displacement():
    """
    PART 2: Plot the RMS displacement for walks between N = [10, 10^6] steps
    """

    num_trials = 2  # Number of trials at each N to be averaged over
    num_lengths = 10  # Number of N values to try
    start_exponent = 1  # i.e. N = 10^1
    end_exponent = 6  # i.e. N = 10^6

    rms_list = []

    walk_lengths = np.logspace(start_exponent, end_exponent, num_lengths, base=10)

    for walk_length in tqdm(walk_lengths):

        # Convert walk_length to an integer number of steps
        n = int(walk_length)

        # r_sum holds the sum of the squared displacements
        r_sum = 0

        for i in range(num_trials):
            _, _, r = random_walk(n)
            r_sum = r_sum + r ** 2

        rms_list.append(np.sqrt(r_sum / num_trials))

    fig, ax = plt.subplots()
    ax.plot(np.log10(walk_lengths), np.log10(rms_list), '-b')
    ax.set_title("Average Displacement as a function of walklength")
    ax.set_xlabel("log10 Walklength (N steps)")
    ax.set_ylabel("log10 RMS Displacement")
    fig.show()
    # fig.savefig("./figs/2.png", dpi=300)


def main():
    random_walk_100_steps()
    find_rms_displacement()


if __name__ == "__main__":
    main()
