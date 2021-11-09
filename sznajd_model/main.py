import matplotlib.pyplot as plt
import numpy as np


def find_magnetisation(row):
    return np.average(row)


def plot_groupthink():

    # Parameters of sim
    nslots  = 40
    nsteps  = 4_000
    f       = 0.25
    s_vals  = [+1, -1]
    probs   = [f, 1 - f]
    output_sampling = 30

    # Preallocate space
    np.random.seed(1)
    arr = np.zeros((nsteps, nslots))
    mag = np.zeros(nsteps)

    # Initialise the first row
    arr[0, :] = np.random.choice(s_vals, size=nslots, p=probs)
    mag[0] = find_magnetisation(arr[0, :])

    for step in range(1, nsteps):

        prev_row = arr[step - 1, :]

        # Select entry
        i = np.random.randint(0, nslots)
        selection = np.take(prev_row, range(i - 1, i + 3), mode='wrap')

        # Rules
        j = 1
        if selection[j] == selection[j + 1]:
            selection[j - 1] = selection[j]
            selection[j + 2] = selection[j]
        else:
            selection[j - 1] = selection[j + 1]
            selection[j + 2] = selection[j]

        # Place back in the array
        arr[step, :] = prev_row[:]
        np.put(arr[step, :], range(i - 1, i + 3), selection, mode='wrapped')
        mag[step] = find_magnetisation(arr[step, :])

    # plot a subset of the steps
    sampled_array = arr[::output_sampling]
    fig, ax = plt.subplots()
    ax.imshow(sampled_array.T, interpolation="none", origin="upper")
    ax.set_title(f"Sampled states, n = {output_sampling}")
    ax.set_xlabel(f"Time [step / {output_sampling}]")
    ax.set_aspect('equal', adjustable='box')
    fig.show()
    # fig.savefig("./figs/1.png")

    # Plot the progress of the magnetisation
    fig2, ax2 = plt.subplots()
    ax2.plot(mag)
    ax2.set_title(f"Net magnetisation, states = {s_vals}")
    ax2.set_xlabel("Time [step]")
    ax2.set_ylabel("Magnetisation")
    fig2.show()
    # fig2.savefig("./figs/2.png")


def main():
    plot_groupthink()


if __name__ == "__main__":
    main()
