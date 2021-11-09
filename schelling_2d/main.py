import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np

# Helper Functions
EMP = 0
BLU = 1
RED = 2

cols = ['white', 'blue', 'red']
cmap = mplc.ListedColormap(cols)


def find_familiarity_fraction(house_colour, neighbour_dict):
    """
    FindFamiliarityFraction finds the fraction of neighbours that
    are the same colour as a household. neighbourDict is a dictionary
    whose keys are the EMP, BLU and RED with corresponding values as
    the number of neighbours of that colour. houseColour can be BLU or
    RED.
    """

    b = neighbour_dict.get(BLU, EMP)
    r = neighbour_dict.get(RED, EMP)

    if house_colour == BLU:
        f = b / (r + b)
    elif house_colour == RED:
        f = r / (r + b)
    else:
        raise ValueError(f"Unknown house_colour {house_colour}")

    return f


def find_diversity_score(house_colour, neighbour_dict):
    """
    FindDiversityScore returns the fraciton of neighbours of a different
    colour to a household, including empty neighbours.

    For example, if a red household has one blue neighbour and seven empty
    plots, its diversity score is 1/8.
    """

    b = neighbour_dict.get(BLU, EMP)
    r = neighbour_dict.get(RED, EMP)

    if house_colour == BLU:
        ds = r / 8
    elif house_colour == RED:
        ds = b / 8
    else:
        raise ValueError

    return ds


def get_neighbour_dict(array, row, col):
    """
    GetNeighbourDict returns a dictionary whose vals are the number of neighbours
    of a particular colour
    """

    house_colour = array[row, col]

    # Find the composition of the neighbourhood
    neighbour_rows = range(row - 1, row + 2)
    neighbour_cols = range(col - 1, col + 2)
    sub_array = array.take(neighbour_rows, mode='wrap', axis=0).take(neighbour_cols, mode='wrap', axis=1)

    unique, counts = np.unique(sub_array, return_counts=True)
    neighbour_dict = dict(zip(unique, counts))

    # Make sure that the house itself does not appear in the dict, only the neighbours
    neighbour_dict[house_colour] = neighbour_dict[house_colour] - 1

    return neighbour_dict


def move_homes_around(array, s):
    """
    move_homes_around loopers over the array in a while loop and moves households to
    different locations if their number of like-coloured neighbours is too low. The
    loop quits when no households move anymore. Returns the array and also a 'diversity'
    and 'familiarity' array, as defined by the problem statement.
    """

    # Preinitialise the familiarity/diversity arrays
    n = array.shape[0]
    familiarity_array = np.zeros((n, n))
    diversity_array = np.zeros((n, n))
    moves = 1

    # Continually perform this process until moves = 0. That is, until no more houses move
    while moves > 0:

        # Reset number of moves
        moves = 0

        # For all households, find the composition of the neighbourhood and move if necessary
        for row in range(0, n):
            for col in range(0, n):

                # Find the household colour
                house_colour = array[row, col]

                if house_colour != EMP:

                    # Find the composition of the neighbourhood
                    neighbour_dict = get_neighbour_dict(array, row, col)

                    # Find the familiarity fraction and diversity score
                    f = find_familiarity_fraction(house_colour, neighbour_dict)
                    familiarity_array[row, col] = f

                    ds = find_diversity_score(house_colour, neighbour_dict)
                    diversity_array[row, col] = ds

                    if f < s:  # move
                        moves += 1

                        x, y = np.where(array == 0)
                        # we chose one index randomly
                        i = np.random.randint(len(x))
                        array[x[i], y[i]] = house_colour
                        array[row, col] = 0

    return array, familiarity_array, diversity_array


def plot_segregation():
    # PART 1: Plot the segregation process

    # Make the results reproducible by specifying the seed
    np.random.seed(0)

    n = 30  # side length of array
    s = 0.3  # similarity

    # Set the proportion of blue and red households
    b = 0.4  # Fraction of blues
    r = 0.4  # Fraction of reds
    w = 1 - b - r  # Fraction of whites / empty houses

    # Randomly initialise the array
    array_before = np.random.choice([EMP, BLU, RED], size=(n, n), p=[w, b, r])

    # Move homes around until everyone is satisfied
    array_after, _, _ = move_homes_around(array_before.copy(), s)

    # Plot results
    fig, ax = plt.subplots(1, 2, sharex='all')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.suptitle("Segregated cities | Before and after")
    ax[0].imshow(array_before, interpolation='none', cmap=cmap)
    ax[0].set_title("Before")
    ax[1].imshow(array_after, interpolation='none', cmap=cmap)
    ax[1].set_title("After")
    fig.show()
    # fig.savefig("./figs/1.png", dpi=300)


def neighbourhood_diversity():

    # PART 2 and PART 3: Diverse neighbourhoods

    n = 30  # side length of array

    # Set the proportion of blue and red households
    b = 0.4  # Fraction of blues
    r = 0.4  # Fraction of reds
    w = 1 - b - r  # Fraction of whites / empty houses

    # For a range of different similarity values, find the average number of diverse neighbours
    # (diversity Count) and the average fraction of neighbours of a different colour
    num_averages = 4
    similarity_vals = np.linspace(0, 0.8, 10)
    diversity_vals = []
    diversity_frac = []

    for s in similarity_vals:
        d_count = 0
        f_count = 0

        for trial in range(0, num_averages):
            array_before = np.random.choice([EMP, BLU, RED], size=(n, n), p=[w, b, r])
            _, familiarity_array, diversity_array = move_homes_around(array_before.copy(), s)

            # Find the proportion that have at least one differently-coloured neighbour
            # (i.e. houses for whom f < 1)

            d_count += np.count_nonzero(familiarity_array < 1)
            f_count += np.mean(diversity_array)

        # Take the average
        diversity_vals.append(d_count / num_averages)
        diversity_frac.append(f_count / num_averages)

    # Plot the results
    fig1, ax1 = plt.subplots()
    ax1.plot(similarity_vals, diversity_vals)
    ax1.set_xlabel("Similarity")
    ax1.set_ylabel("Number of homes")
    ax1.set_title(f"Number of homes with at least one diverse neighbour, grid = ({n} x {n})")
    fig1.show()
    # fig1.savefig("./figs/2.png", dpi=300)

    fig2, ax2 = plt.subplots()
    ax2.plot(similarity_vals, diversity_frac)
    ax2.set_xlabel("Similarity")
    ax2.set_ylabel("Diversity fraction")
    ax2.set_title(f"Fraction of neighbours of opposite colour, grid = ({n} x {n})")
    fig2.show()
    # fig2.savefig("./figs/3.png", dpi=300)


def main():
    plot_segregation()
    neighbourhood_diversity()


if __name__ == "__main__":
    main()
