from scipy.stats import linregress
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random


class Aggregator:

    def __init__(self, n, seed):
        self.n = n
        self.arr = np.zeros((n, n))
        self.r_vals = []
        self.points = []
        self.a_sites = []
        self.seed = seed
        self.min_x = seed[0]
        self.max_x = seed[0]
        self.min_y = seed[1]
        self.max_y = seed[1]
        self.add_point(seed)

    def is_agg_point(self, pt):

        ptx, pty = pt

        if ptx < self.min_x or ptx > self.max_x or pty < self.min_y or pty > self.max_y:
            return False
        else:
            return pt in self.a_sites

    def is_point(self, pt):
        ptx, pty = pt

        if ptx < self.min_x or ptx > self.max_x or pty < self.min_y or pty > self.max_y:
            return False
        else:
            return pt in self.points

    def add_agg_point(self, pt):
        self.a_sites.append(pt)

        ptx, pty = pt

        if ptx < self.min_x:
            self.min_x = ptx
        if ptx > self.max_x:
            self.max_x = ptx
        if pty < self.min_y:
            self.min_y = pty
        if pty > self.max_y:
            self.max_y = pty

    def add_point(self, pt):

        if not self.is_point(pt):

            self.points.append(pt)
            self.r_vals.append(self.estimate_r())
            self.arr[pt] = 1

            if self.is_agg_point(pt):
                self.a_sites.remove(pt)

            # Check the aggregation points around the added point
            px, py  = pt
            left    = (px - 1, py)
            right   = (px + 1, py)
            up      = (px, py + 1)
            down    = (px, py - 1)

            if not self.is_agg_point(left) and not self.is_point(left):
                self.add_agg_point(left)
            if not self.is_agg_point(right) and not self.is_point(right):
                self.add_agg_point(right)
            if not self.is_agg_point(up) and not self.is_point(up):
                self.add_agg_point(up)
            if not self.is_agg_point(down) and not self.is_point(down):
                self.add_agg_point(down)

    def estimate_r(self):
        pts = np.array(self.points)
        origin = np.array(self.seed)
        tmp = int(np.ceil(np.max(np.apply_along_axis(np.linalg.norm, 1, pts - origin))))
        return max(tmp, 1)

    def plot_r(self):
        return max(self.estimate_r(), 4)

    def plot(self):
        r = int(1.1 * self.plot_r())
        sdx, sdy = self.seed
        extent = [sdx-r, sdx+r, sdy+r, sdy-r]
        fig, ax = plt.subplots()
        ax.imshow(self.arr[sdx-r:sdx+r, sdy-r:sdy+r], interpolation="none", origin="upper",
                  extent=extent)
        ax.set_title("Diffusion Limited Aggregation")
        fig.show()
        # fig.savefig("./figs/1.png")

    def plot_radial_growth(self):

        # Linear regression of log data
        num_agg_points = range(1, len(self.r_vals) + 1)
        lgx = np.log10(num_agg_points)
        lgy = np.log10(self.r_vals)
        slope, intercept, r_value, _, _ = linregress(lgx, lgy)

        fig, ax = plt.subplots()
        ax.plot(lgx, lgy, "k.", label="R(N)")
        plt.plot(lgx, lgx * slope + intercept, label="Linear fit")
        ax.set_xlabel("log10(N)")
        ax.set_ylabel("log10(R)")
        ax.set_title(f"Growth Rate of crystal: w = {round(slope, 2)}")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.legend()
        fig.show()
        # fig.savefig("./figs/2.png")

    def get_launch_location(self):
        pr = self.plot_r()
        th = 2 * np.pi * np.random.rand()
        lx = int(self.seed[0] + pr * np.cos(th))
        ly = int(self.seed[1] + pr * np.sin(th))
        return lx + 3, ly + 3

    def aggregate_point(self):
        diffusion_limit = 3 * self.plot_r()
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        lx, ly = self.get_launch_location()

        # Perform a random walk
        x, y = lx, ly
        while not self.is_agg_point((x, y)):
            dx, dy = random.choice(offsets)
            x, y = x + dx, y + dy

            # If outside of bounds, discard trial solution
            if x < 0 or x > self.n or y < 0 or y > self.n:
                break

            if np.sqrt((x - lx) ** 2 + (y - ly) ** 2) > diffusion_limit:
                break

        if self.is_agg_point((x, y)):
            self.add_point((x, y))


def create_aggregated_object():

    np.random.seed(0)
    n           = 1000
    seed        = (500, 500)
    n_trials    = 4000

    # Create and aggregate the object
    agg = Aggregator(n, seed)
    for _ in tqdm(range(n_trials)):
        agg.aggregate_point()

    print(f"Percentage of trial points added: {round(len(agg.points) / n_trials, 2) * 100} %")

    agg.plot()
    agg.plot_radial_growth()


def main():
    create_aggregated_object()


if __name__ == "__main__":
    main()
