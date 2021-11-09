import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 6})


def potential(x, a):
    """
    Defines the potential function at a particular x or array of x's
    """
    return - np.power(x, 2) + a * np.power(x, 4)


def d_potential(x, a):
    """
    Defines the derivative of the potential at a particular x or array of x's
    """
    return - 2 * x + 4 * a * np.power(x, 3)


def update_x(x_old, dt, xi, kt, a):
    """
    Performs a time update on the state xOld, returning x_new
    """

    del_d_del_x = 0  # since kT and xi are constant
    del_potential = d_potential(x_old, a)

    # Generate the random term, g(t)
    mu = 0
    sig = np.sqrt(2 * kt * 1 / xi * dt)
    g = np.random.normal(mu, sig)

    x_new = x_old - 1 / xi * del_potential * dt + g + del_d_del_x * dt

    return x_new


def get_parameters_from_barrier_height(e):
    """ Finds the value of alpha and the bottom of the left well (xStart) from the barrier height, e"""

    a       = 1 / (4 * e)
    x_start = - np.sqrt(1 / (2 * a))

    return x_start, a


def langevin_simulation(e, xi, kt, dt, t_max):
    """
    Performs an Inertialless Langevin Simulation of a particle starting at x_start in a potential defined by
    potential(x) from time [0, tMax]. The function returns a vector of t (time), x (position), u (potential) values
    as well as the a ( alpha) value used and the amount of time before the particle first crossed over into the right
    well (if no crossing was made, crossover_time = None). Input is the barrier height, E """

    x_start, a = get_parameters_from_barrier_height(e)
    time = 0
    crossover_time = None

    # Initialise vectors to hold the trajectories
    t = [time]
    x = [x_start]
    u = [potential(x_start, a)]
    x_curr = x_start

    while time < t_max:
        x_curr = update_x(x_curr, dt, xi, kt, a)

        x.append(x_curr)
        t.append(time)
        u.append(potential(x_curr, a))

        # Record the first crossover into the right well
        if x_curr > -x_start and crossover_time is None:
            crossover_time = time

        time += dt

    return t, x, u, a, crossover_time


def plot_typical_trajectories():
    # PART 1: Plotting typical trajectories
    kt = 1
    xi = 1
    dt = 0.1
    t_max = 25

    # Make the results reproducible by specifying the seed
    np.random.seed(0)

    barrier_heights = [0.25, 0.50, 0.75, 1.00, 2.00, 3.00]

    # Set up a figure to old all the results
    fig, ax = plt.subplots(nrows=len(barrier_heights), ncols=2, sharex='col')
    plt.tight_layout()

    for i in range(0, len(barrier_heights)):
        # Set the current barrier height
        e = barrier_heights[i]

        # Conduct a simulation at the barrier height, e
        t, x, u, a, _ = langevin_simulation(e, xi, kt, dt, t_max)

        # Plot results
        subplot_title = f"a = {round(a, 2)}. e = {e}"
        ax[i, 0].plot(t, x)
        ax[i, 0].set_title(subplot_title)
        ax[i, 0].set_ylabel("x(t)")

        x_plot = np.linspace(-4, 4, 100)
        ax[i, 1].plot(x_plot, potential(x_plot, a), label='Potential')
        ax[i, 1].plot(x, u, 'r.', label='Particle behaviour')
        ax[i, 1].set_title(subplot_title)
        ax[i, 1].set_ylabel("U(x)")
        ax[i, 1].legend()

    ax[len(barrier_heights) - 1, 0].set_xlabel("Time / s")
    ax[len(barrier_heights) - 1, 1].set_xlabel("x(t)")
    plt.subplots_adjust(left=0.125, hspace=0.5, top=0.9, wspace=0.25, bottom=0.1)
    fig.suptitle("Inertialess Langevin Simulation, potential(x) = -x^2 + alpha x^4")
    fig.show()
    # fig.savefig("./figs/1.png", dpi=300)


def plot_crossing_times():
    # PART 2: Time needed to cross to the other side
    kt = 1
    xi = 1
    dt = 0.1
    t_max = 100
    num_averages = 20

    # Measure the average crossover time over 10 trials
    barrier_heights = np.linspace(0.25, 5, 50)
    crossover_avgs = []

    for i in range(0, len(barrier_heights)):

        # Set the current barrier height
        e = barrier_heights[i]

        # Conduct num_trials simulations and record the average crossover time
        crossover_counter = 0
        num_trials = num_averages

        for j in range(0, num_averages):
            _, _, _, _, crossover_time = langevin_simulation(e, xi, kt, dt, t_max)

            # If particle never crossed over, ignore that trial
            if crossover_time is not None:
                crossover_counter += crossover_time
            else:
                num_trials -= 1

        crossover_avgs.append(crossover_counter / num_trials)

    fig, ax = plt.subplots()
    ax.plot(barrier_heights, np.log(crossover_avgs))
    ax.set_title("Crossover Time as a function of barrier height, averaged over " + str(num_averages) + "trials")
    ax.set_xlabel("Barrier height, e (Energy units)")
    ax.set_ylabel("Log(Crossover Time)")
    fig.show()
    # fig.savefig("./figs/2.png", dpi=300)


def main():
    plot_typical_trajectories()
    plot_crossing_times()


if __name__ == "__main__":
    main()
