import numpy as np
import matplotlib.pyplot as plt


def spin_energy(spin):
    """
    spin_energy() returns the energy of the system as a function of the spin
    """
    return spin


def metropolis_monte_carlo(x_start, kt, n):
    """
    metropolis_monte_carlo() conducts a Metropolis Monte Carlo simulation over N timesteps,
    starting at the state xStart with Boltzmann Energy kT. This function returns a list
    of the value of the running average across each of the timesteps
    """

    # Step 0: Set first state
    state = x_start
    state_list = [state]
    run_avgs = [state]

    for i in range(1, n):

        # Step 1: Find a trial state (for a two level system, this is just the opposite state)
        trial_state = not state

        # Step 2: Find the energy difference
        d_energy = spin_energy(trial_state) - spin_energy(state)

        # Step 3: If energy decreases, accept as new state, else only accept it based on a probabilistic jump
        if d_energy < 0:
            state = trial_state
        else:
            if np.random.random() < np.exp(-d_energy / kt):
                state = trial_state

        # Keep the running average
        run_avgs.append((i * run_avgs[-1] + state) / (i + 1))
        state_list.append(state)

    return run_avgs, state_list


def kt_09():
    # PART 1:

    # Make the results reproducible by specifying the seed
    np.random.seed(0)

    # Starting variables
    x_start = 0
    n       = 100
    steps   = range(0, n)
    kt      = 0.9

    # Simulate
    run_avgs, state_list = metropolis_monte_carlo(x_start, kt, n)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(steps, state_list, '-b', label='Spin')
    ax.plot(steps, run_avgs, '-r', label='Running Average')
    ax.set_xlabel("Step number")
    ax.set_ylabel("Spin State / Running Average")
    ax.set_title(f"Monte Carlo Simulation for two spin states, {kt=}")
    ax.legend()
    fig.show()
    # fig.savefig(f"./figs/{x_start=}_{n=}_{kt=}.png", dpi=300)
    

def kt_11():

    # Make the results reproducible by specifying the seed
    np.random.seed(0)

    # Simulation for kT = 1.1
    x_start = 0
    n       = 100
    steps   = range(0, n)
    kt      = 1.1

    # Simulate
    run_avgs, state_list = metropolis_monte_carlo(x_start, kt, n)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(steps, state_list, '-b', label='Spin')
    ax.plot(steps, run_avgs, '-r', label='Running Average')
    ax.set_xlabel("Step number")
    ax.set_ylabel("Spin State / Running Average")
    ax.set_title(f"Monte Carlo Simulation for two spin states, {kt=}")
    ax.legend()
    fig.show()
    # fig.savefig(f"./figs/{x_start=}_{n=}_{kt=}.png", dpi=300)


def main():
    kt_09()
    kt_11()


if __name__ == "__main__":
    main()
