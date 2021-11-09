import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def advance_timestep(sub_array):
    """
    advance_timestep() This function takes a numpy array of size (1,3) containing ones and zeros,
    where a one represents a car in a slot and zero represents the absence of a car
    """

    # Cast the subArray as a string so we can match it to the keys in switcher
    sub_array_string = np.array2string(sub_array.astype(int))

    # switcher is a psuedo 'switch' statement that lets us select which patterns
    # map to a new car at the central index
    switcher = {"[1 1 1]": 1,
                "[1 0 1]": 1,
                "[1 0 0]": 1,
                "[0 1 1]": 1}

    # switcher.get() returns zero if sub_array_string doesn't match one of the above
    # patterns
    return switcher.get(sub_array_string, 0)


def simulate_traffic(nstps, n_spaces, num_cars):
    """
    simulate_traffic performs a traffic simulation on a 1d road containing numCars and n_spaces over
    nstps timesteps. The cars are assigned randomly on the first timestep. Car positions over time are
    stored in the array, where a 1 indicates a car. The timestep is advanced by taking a subarray of size
    (1,3) and assigning to the row below using the rules enumerated in the advance_timestep() function.
    Speed is measured by the average number of cars that move on a particular step divided by the number
    of cars in total. The grid has periodic boundaries on its sides.
    """

    # All data will be stored in the variable array, which will be plotted to display car motion
    # Preinitialise array's first row randomly
    array = np.zeros((nstps, n_spaces))
    array[0, :] = np.random.choice([0, 1], size=n_spaces, p=[1 - num_cars / n_spaces, num_cars / n_spaces])

    # Preinitialise an array to hold the speeds over the course of the simulation
    avg_speeds = np.zeros(nstps - 1)

    for row in range(1, nstps):

        # prev_state is the row above the current array row. num_cars_moved counts how many cars advance
        num_cars_moved = 0
        prev_state = array[row - 1, :]

        for col in range(0, n_spaces):

            # Extract a three-element array from the previous row and update the element [row, col]
            # and advance the timestep
            sub_array = prev_state.take(range(col - 1, col + 2), mode='wrap')
            new_state = advance_timestep(sub_array)
            array[row, col] = new_state

            # Count any cars that moved
            car_moved = (new_state == 1) and (sub_array[1] == 0)
            if car_moved:
                num_cars_moved += 1

        # Find the average movement speed this timestep
        avg_speeds[row - 1] = num_cars_moved / num_cars

    return array, np.mean(avg_speeds)


def plot_pattern_and_density():
    """TASK 1: Display pattern for different densities"""

    # Make the results reproducible by specifying the seed
    np.random.seed(0)

    n_steps = 100
    n_car_spaces = 100
    num_cars_light = 25
    num_cars_medium = 50
    num_cars_heavy = 75
    num_cars_full = 95

    # Simulate traffic for four different levels of crowded-ness
    array1, _ = simulate_traffic(n_steps, n_car_spaces, num_cars_light)
    array2, _ = simulate_traffic(n_steps, n_car_spaces, num_cars_medium)
    array3, _ = simulate_traffic(n_steps, n_car_spaces, num_cars_heavy)
    array4, _ = simulate_traffic(n_steps, n_car_spaces, num_cars_full)

    # Display and save the output
    fig, ax = plt.subplots(2, 2, sharex='all')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle("1D Traffic flow with cars in yellow. (n_steps, n_car_spaces, numCars) = ")
    ax[0, 0].imshow(array1)
    ax[0, 0].set_title(f"({n_steps}, {n_car_spaces}, {num_cars_light})")
    ax[0, 1].imshow(array2)
    ax[0, 1].set_title(f"({n_steps}, {n_car_spaces}, {num_cars_medium})")
    ax[1, 0].imshow(array3)
    ax[1, 0].set_title(f"({n_steps}, {n_car_spaces}, {num_cars_heavy})")
    ax[1, 1].imshow(array4)
    ax[1, 1].set_title(f"({n_steps}, {n_car_spaces}, {num_cars_full})")
    fig.show()
    # fig.savefig("./figs/1.png", dpi=300)


def plot_speed_and_density():
    """TASK 2: Speed and density"""

    # Find the average speed in the range from density 0 to 1
    n_car_spaces = 100
    n_steps = 10
    num_avgs = 1
    num_cars_to_simulate = np.arange(10, n_car_spaces, 1)
    lin_density = num_cars_to_simulate / n_car_spaces
    avg_car_speed = np.zeros(len(num_cars_to_simulate))

    # Simulate traffic for different densities
    for idx, num_cars in tqdm(enumerate(num_cars_to_simulate), total=len(num_cars_to_simulate)):

        # Find the average speed over num_avgs trials
        speed_counter = 0

        for i in range(num_avgs):
            _, speed = simulate_traffic(n_steps, n_car_spaces, num_cars)
            speed_counter += speed

        avg_car_speed[idx] = speed_counter / num_avgs

    # Show or export figure
    fig, ax = plt.subplots()
    ax.plot(lin_density, avg_car_speed)
    ax.set_title("Simulated car speed as a function of car density")
    ax.set_xlabel("Car density (number of cars / number of spaces)")
    ax.set_ylabel("Car speed (arbitrary units)")
    fig.show()
    # fig.savefig("./figs/2.png", dpi=300)


def main():
    plot_pattern_and_density()
    plot_speed_and_density()


if __name__ == "__main__":
    main()
