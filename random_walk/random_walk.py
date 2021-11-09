import numpy as np


def theta():
    """
    theta() returns a random angle between [0, 2pi) in radians
    """
    return 2 * np.pi * np.random.random()


def random_walk(n, return_all_steps=False):
    """
    random_walk conducts a random walk of n steps in the 2D plane starting at (x,y) = (0,0). The step size is lambda = 1
    return_all_steps is a boolean that will return the x,y locations of all steps as a list if set to True.
    """

    # Walk parameters
    x0 = 0
    y0 = 0
    step_size = 1

    # Initialise arrays to store the data
    x_locations = np.zeros(n)
    y_locations = np.zeros(n)

    x_locations[0] = x0
    y_locations[0] = y0

    for step_number in range(1, n):
        # Find a random direction to walk in
        th = theta()

        # Find the resulting change to the position
        dx = step_size * np.cos(th)
        dy = step_size * np.sin(th)

        # Append to the list of locations
        x_locations[step_number] = x_locations[step_number - 1] + dx
        y_locations[step_number] = y_locations[step_number - 1] + dy

    # Find the total distance travelled from the start, R
    r = np.sqrt((x_locations[-1] - x0) ** 2 + (y_locations[-1] - y0) ** 2)

    if return_all_steps:
        return x_locations, y_locations, r
    else:
        return x_locations[-1], y_locations[-1], r
