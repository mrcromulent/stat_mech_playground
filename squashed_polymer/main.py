import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def random_walk(n, d):
    """
    random_walk conducts a random walk of n steps in the 2D plane starting at (x,y) = (0,0). The step size is lambda = 1
    return_all_steps is a boolean that will return the x,y locations of all steps as a list if set to True.
    """

    # Walk parameters
    x0 = 0
    y0 = 0
    step_size = 1
    ylim = d / 2

    # Initialise arrays to store the data
    locations = np.zeros((2, n))
    locations[0, 0] = x0
    locations[1, 0] = y0

    s = 1
    wall_intersect = False
    for stp_num in range(1, n):
        # Find a random direction to walk in
        th = 2 * np.pi * np.random.random()

        # Find the resulting change to the position
        dx = step_size * np.cos(th)
        dy = step_size * np.sin(th)

        # Append to the list of locations
        s = stp_num
        locations[0, s] = locations[0, s - 1] + dx
        locations[1, s] = locations[1, s - 1] + dy

        if locations[1, s] > ylim:
            wall_intersect = True
            break

    # Find the total distance travelled from the start, r
    r = np.sqrt((locations[0, s] - x0) ** 2 + (locations[1, s] - y0) ** 2)

    return locations, r, wall_intersect


def plot_run(n, d, ax, seed):
    np.random.seed(seed)
    locs, r, wall_intersect = random_walk(n, d)
    ax.plot(locs[0, :], locs[1, :])
    ax.axhline(y=- d / 2, color='r')
    ax.axhline(y=+ d / 2, color='r')
    ax.set_title(f"{d=}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_aspect('equal')


def plot_typical_states():

    n = 10_000

    fig, ax = plt.subplots(2, 2)

    # Maximum distance
    d = 2 * n
    plot_run(n, d, ax[0, 0], 0)

    # Moderate distance
    d = 0.5 * n
    plot_run(n, d, ax[0, 1], 0)

    # Small distance
    d = 0.05 * n
    plot_run(n, d, ax[1, 0], 0)

    # Tiny distance
    d = 0.01 * n
    plot_run(n, d, ax[1, 1], 4)

    fig.tight_layout()
    fig.show()
    # fig.savefig("./figs/1.png")


def plot_free_energy_v_force():

    np.random.seed(0)

    q = 10_000
    kt = 1
    n = 50

    dmax = 2 * n
    dmin = 1
    num_ds = 50
    # d_vals = np.logspace(np.log10(dmax), np.log10(dmin), num=num_ds)
    d_vals = np.linspace(dmax, dmin, num=num_ds)

    n_d = np.zeros(num_ds)

    for i, d in enumerate(d_vals):
        for j in tqdm(range(q)):
            lcs, r, wall_intersect = random_walk(n, d)
            if not wall_intersect:
                n_d[i] += 1

    free_energy = - kt * np.log(n_d)
    force = - np.diff(free_energy) / np.diff(d_vals)
    force = np.hstack((force[0], force))

    fig, ax = plt.subplots()
    ax.loglog(-free_energy, force)
    ax.set_title("Force on plates from polymer")
    ax.set_xlabel("abs(Free energy (F))")
    ax.set_ylabel("Force (f)")
    fig.show()
    # fig.savefig("./figs/2.png")


def main():

    plot_typical_states()
    plot_free_energy_v_force()


if __name__ == "__main__":
    main()
