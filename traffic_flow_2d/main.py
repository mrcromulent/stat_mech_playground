import matplotlib.animation as ani
import matplotlib.colors as mplc
import matplotlib.pyplot as plt
from tqdm import tqdm
from enum import Enum
import numpy as np

cols = ['white', 'red', 'blue']
cmap = mplc.ListedColormap(cols)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


class Colour(Enum):
    WHITE = 0
    RED = 1
    BLUE = 2


class TrafficAnimation:
    im = None
    pbar = None
    num_cars = 0
    mobility = []

    def __init__(self, n, cars, nsteps):

        self.n = n
        self.cars = cars
        self.nsteps = nsteps
        self.arr = np.zeros((nsteps, n, n))

        self.init_arr()
        self.init_mobility()
        self.populate_arr()

    def init_mobility(self):
        self.num_cars = np.count_nonzero(self.arr)
        self.mobility = np.zeros(self.nsteps)
        self.mobility[0] = 1.0

    def init_arr(self):
        self.arr[0, :, :] = np.random.choice([v.value for v in Colour.__members__.values()],
                                             size=(self.n, self.n),
                                             p=self.cars)

    def populate_arr(self):

        arr = self.arr
        n = self.n

        for t in range(1, self.nsteps):

            # Copy the previous configuration
            arr[t, :, :] = arr[t - 1, :, :]
            num_moves = 0

            # Move red cars
            if t % 2 == 0:
                red_row, red_col = np.where(arr[t, :, :] == Colour.RED.value)
                for r, c in zip(red_row, red_col):
                    if arr[t, r, (c + 1) % n] == Colour.WHITE.value:
                        arr[t, r, (c + 1) % n] = Colour.RED.value
                        arr[t, r, c] = Colour.WHITE.value
                        num_moves += 1

                self.mobility[t] = num_moves / len(red_row)

            # Move blue cars
            if t % 2 == 1:
                blu_row, blu_col = np.where(arr[t, :, :] == Colour.BLUE.value)
                for r, c in zip(blu_row, blu_col):
                    if arr[t, (r + 1) % n, c] == Colour.WHITE.value:
                        arr[t, (r + 1) % n, c] = Colour.BLUE.value
                        arr[t, r, c] = Colour.WHITE.value
                        num_moves += 1

                self.mobility[t] = num_moves / len(blu_row)

    def show_steps(self, steps):

        nplots = len(steps)
        nrows = 2
        ncols = int(nplots / 2)

        fig, ax = plt.subplots(nrows, ncols, sharey='row')
        for i in range(0, nplots):
            curr_arr = self.arr[steps[i], :, :]
            curr_ax = ax[int(i / (nplots / 2)), int(i % (nplots / 2))]
            curr_ax.imshow(curr_arr, interpolation="none", origin="upper", cmap=cmap)
            curr_ax.set_title(f"Step: {steps[i]}")
        fig.tight_layout()
        fig.show()
        # fig.savefig("./figs/1.png")

    def plot_mobility(self, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.plot(moving_average(self.mobility))
        return fig

    def animation_init(self):
        return [self.im]

    def animate(self, i):
        self.pbar.update(1)
        self.im.set_array(self.arr[i + 1, :, :])
        return [self.im]

    def make_animation(self):

        self.pbar = tqdm(total=self.nsteps - 1)

        fig, ax = plt.subplots()
        self.im = ax.imshow(self.arr[0, :, :], interpolation="none", origin="upper", cmap=cmap)
        ax.set_title("Car motion")
        anim = ani.FuncAnimation(fig,
                                 self.animate,
                                 init_func=self.animation_init,
                                 frames=self.nsteps - 1)
        anim.save('./figs/traffic_flow.mp4', fps=30)


def show_steps():
    np.random.seed(0)
    n = 30
    nsteps = 1000

    b = 0.25
    r = 0.25
    w = 1 - b - r

    ta = TrafficAnimation(n, [w, r, b], nsteps)
    ta.show_steps([int(p * nsteps) for p in np.linspace(0.0, 0.99, 8)])


def create_animation():
    np.random.seed(0)
    n = 30
    nsteps = 1000

    b = 0.25
    r = 0.25
    w = 1 - b - r

    ta = TrafficAnimation(n, [w, r, b], nsteps)
    ta.make_animation()


def plot_mobility():
    np.random.seed(0)
    n = 30
    nsteps = 500

    # Density of traffic of both colours
    density = np.linspace(0.01, 1.00, 10)

    #
    fig, ax = plt.subplots()
    for d in density:

        b = d / 2
        r = d / 2
        w = 1 - b - r
        ta = TrafficAnimation(n, [w, r, b], nsteps)
        fig = ta.plot_mobility(fig=fig, ax=ax)

    ax.set_title("Mobility for different densities")
    ax.set_xlabel("Step [n]")
    ax.set_ylabel("Mobility (percentage of cars able to move)")
    ax.legend([f"r = b = {round(100 * d, 2)} %" for d in density])
    fig.show()
    # fig.savefig("./figs/2.png")


def main():
    show_steps()
    create_animation()
    plot_mobility()


if __name__ == "__main__":
    main()
