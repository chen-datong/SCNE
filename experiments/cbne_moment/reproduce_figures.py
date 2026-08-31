"""Regenerate the two active CBNE moment figures from retained data."""

import os

import matplotlib

matplotlib.use("Agg")

import plot


def main():
    os.makedirs("figures", exist_ok=True)
    plot.plot_Haar_N()
    plot.plot_sample_complexity_lower()


if __name__ == "__main__":
    main()
