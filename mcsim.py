#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "matplotlib"]
# ///

import math
import random
import matplotlib.pyplot as plt
import numpy as np

T = 10.0
N = 1000
steps = 250000

n = np.ones((N, 3), dtype=int)     # array of quantum numbers
energies = []
E = 3 * N * math.pi * math.pi / 2
for k in range(steps):
    i = random.randrange(N)
    j = random.randrange(3)

    if random.random() < 0.5:
        dn = 1
        dE = (2 * n[i, j] + 1) * math.pi * math.pi / 2
    else:
        dn = -1
        dE = (-2 * n[i, j] + 1) * math.pi * math.pi / 2

    if n[i, j] > 1 or dn == 1:
        if random.random() < math.exp(-dE / T):
            n[i, j] += dn
            E += dE

    energies.append(E)


fig, ax = plt.subplots()

ax.plot(energies)
ax.set_ylabel("Energy")

plt.show()

# TODO: example 4.2
# TODO: derivation of monte carlo
