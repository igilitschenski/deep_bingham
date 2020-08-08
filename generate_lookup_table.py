"""
Generates the lookup table for the Binghasm normalization constant.
"""
from __future__ import print_function

import numpy as np
import time
import utils


def generate_bd_lookup_table():
    coords = np.linspace(-500, 0, 40)
    duration = time.time()
    utils.build_bd_lookup_table(
        "uniform", {"coords": coords, "bounds": (-500, 0), "num_points": 40},
        "precomputed/lookup_-500_0_40.dill")
 
    duration = time.time() - duration

    print('lookup table function took %0.3f ms' % (duration * 1000.0))


if __name__ == "__main__":
    generate_bd_lookup_table()
