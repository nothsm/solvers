#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["tinygrad"]
# ///

import math
import random
import time

# from tinygrad import Tensor, TinyJit
# from tinygrad.helpers import tqdm

X_MIN = -16
X_MAX = 16
Y_MIN = -8
Y_MAX = 8
TICK_WIDTH = 3

WIDTH = 7
PRECISION = 2

DATA_SMALL = [[1, 2], [2, 1], [2, 2], [5, 6], [6, 5], [6, 6]]
DATA_MEDIUM = [
    [-1.3532938, 4.3389422],
    [-1.39349887, 4.94316972],
    [-1.9329929, 4.91238801],
    [-0.97237894, 3.62194903],
    [-0.10134372, 4.84366914],
    [-2.11210884, 5.17631355],
    [-1.12743896, 4.40889104],
    [-0.92398987, 5.01162481],
    [-0.8938131, 3.95292412],
    [-0.53634351, 4.42628438],
    [-1.69688431, -2.11571212],
    [-0.82918803, -1.72214341],
    [-2.22918335, -1.33236838],
    [-0.52285403, -2.50249298],
    [-1.1444109, -2.16431946],
    [-0.99925232, -0.92723154],
    [-1.21334868, -2.2518903],
    [-1.03337926, -1.15530737],
    [-1.52382403, -0.57341381],
    [-1.3332308, -2.86994757],
    [1.62477508, 1.31236341],
    [0.99283732, 0.59203994],
    [1.55905108, -0.16178332],
    [1.47259345, 0.8105658],
    [1.38573673, 0.32254359],
    [1.80208398, 0.33238321],
    [0.22232127, 0.24147771],
    [1.23306371, 0.20317336],
    [1.81298418, 1.28993021],
    [0.65879339, -0.1012351]
]

DATASET_FILENAME = "kmeans/data/small.txt"

RED = "\033[91m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
RESET = "\033[0m"
# CLASSES = ['x', 'o', 'w', 'v', 'u','c', 's' '+', '*', '@'] # TODO: add more classes
# CLUSTERS = ['X', 'O', 'W', 'V', 'U', 'C', 'S']

CLASSES = ['r', 'b', 'g', 'y', 'm','w', 's' '+', '*', '@'] # TODO: add more classes
CLUSTERS = ['R', 'B', 'G', 'Y', 'M', 'W', 'S']
COLORS = [RED, BLUE, GREEN, YELLOW, MAGENTA, WHITE]

def distance(xs, ys):
    return 0.5 * sum((y - x)**2 for (x, y) in zip(xs, ys))

# TODO: vectorize this
def distance_matrix_from(xs):
    distances = [[0 for _ in range(len(xs))] for _ in range(len(xs))]
    for (i, x) in enumerate(xs):
        for (j, y) in enumerate(xs):
            distances[i][j] = distance(x, y)
    return distances

# TODO: KLUDGE ALERT!
def distance_matrix_display(xss):
    assert(WIDTH % 2 == 1)
    n_rows = len(xss)
    n_cols = len(xss[0])

    print()
    for j in range(n_cols):
        if j == 0:
            print(" " * WIDTH, end="|")
        x_str = ""
        if len(str(j)) == 1:
            x_str = f"  x{j}   " # TODO: this does not scale
        else:
            x_str = f" x{j}   "
        assert len(x_str) == WIDTH, f"len(x_str) = {len(x_str)}"
        print(x_str, end="")
        if j != n_cols - 1:
            print("|", end="")
    print()

    for j in range(n_cols + 1):
        divider = "-" * WIDTH
        if j != n_cols - 1:
            divider = divider + "-"
        print(divider, end="")
    print()

    for i in range(n_rows):
        x_str = ""
        if len(str(j)) == 1:
            x_str = f"  x{j}   " # TODO: this does not scale
        else:
            x_str = f" x{j}   "
        assert len(x_str) == WIDTH, f"len(x_str) = {len(x_str)}"
        print(x_str, end="")
        print("|", end="")
        for j in range(n_cols):
            x = xss[i][j]
            x_str = f"{x:.{PRECISION}f}" # TODO: uglyyyyyy
            if len(x_str) == 4: # TODO: generalize
                x_str = " " + x_str + "  "
            elif len(x_str) == 5:
                x_str = " " + x_str + " "
            else:
                raise Exception(f"kmean: unexpected length for x_str (x_str = {x_str}, len(x_str) = {len(x_str)})")
            assert(len(x_str) == WIDTH)

            print("{}".format(x_str), end="")
            if j != n_cols - 1:
                print("|", end = "")
        print()

        if i != n_rows - 1:
            for j in range(n_cols + 1):
                divider = "-" * WIDTH
                if j != n_cols - 1:
                    divider = divider + "-"
                print(divider, end="")
            print()

# TODO: Add support for floats
def dataset_parse(s, sep=' '):
    def parse1(s):
        try:
            return int(s)
        except ValueError:
            return float(s)
        except ValueError:
            return s
    return [[parse1(word) for word in line.split(sep)] for line in s.strip().split('\n')]

# TODO: this can only plot integral pairs in [-10, 10] x [-10, 10]
# TODO: this can be improved a LOT
def dataset_plot(dataset):
    assert(X_MIN <= X_MAX)
    assert(Y_MIN <= Y_MAX)
    print()
    for y in range(Y_MAX, Y_MIN - 1, -1):
        for x in range(X_MIN, X_MAX):
            is_blank = True # TODO: change this to be positive rather than negative
            for (xi, yi) in dataset:
                if xi == x and yi == y:
                    is_blank = False
                    break
            if x == X_MIN:
                tick = str(y)
                assert(len(tick) == 1 or len(tick) == 2 or len(tick) == 3)
                if len(tick) == 1:
                    tick = " " + tick + " "
                elif len(tick) == 2:
                    tick = tick + " "
                assert(len(tick) == TICK_WIDTH)
                print(tick, end="")
            if is_blank:
                print(".", end="")
            else:
                print(CLASSES[0], end="")

        print()
    # print(" " * TICK_WIDTH, end="")
    for x in range(X_MIN, X_MAX):
        tick = str(x)
        if len(tick) == 1:
            print(f"{x}", end="")
        else:
            print(" ", end="")
    print()

# TODO: repeating code...
def supervised_dataset_plot(dataset):
    assert(X_MIN <= X_MAX)
    assert(Y_MIN <= Y_MAX)
    print()
    for y in range(Y_MAX, Y_MIN - 1, -1):
        for x in range(X_MIN, X_MAX):
            is_blank = True # TODO: change this to be positive rather than negative
            for ((sample_x, sample_y), sample_class) in dataset:
                if sample_x == x and sample_y == y:
                    is_blank = False
                    break
            if x == X_MIN:
                tick = str(y)
                assert(len(tick) == 1 or len(tick) == 2 or len(tick) == 3)
                if len(tick) == 1:
                    tick = " " + tick + " "
                elif len(tick) == 2:
                    tick = tick + " "
                assert(len(tick) == TICK_WIDTH)
                print(tick, end="")
            if is_blank:
                print(".", end="")
            else:
                print(CLASSES[sample_class], end="")

        print()
    print(" " * TICK_WIDTH, end="")
    for x in range(X_MIN, X_MAX):
        tick = str(x)
        if len(tick) == 1:
            print(f"{x}", end="")
        else:
            print(" ", end="")
    print()

# TODO: repeating code again...
def kmeans_dataset_plot(dataset, means, dataset_mean_assignments):
    assert(X_MIN <= X_MAX)
    assert(Y_MIN <= Y_MAX)

    means_seen_len =0
    print()
    for y in range(Y_MAX, Y_MIN - 1, -1):
        for x in range(X_MIN, X_MAX):
            # is this square is filled?
            is_sample = False # TODO: change this to be positive rather than negative
            is_mean = False
            square_class = None
            for ((sample_x, sample_y), sample_mean) in zip(dataset, dataset_mean_assignments):
                if round(sample_x) == x and round(sample_y) == y:
                    is_sample = True
                    square_class = sample_mean
                    break
            for (mean_ix, (mean_x, mean_y)) in enumerate(means):
                if round(mean_x) == x and round(mean_y) == y:
                    is_mean = True
                    square_class = mean_ix
                    means_seen_len += 1
            # if first square, print y tick
            if x == X_MIN:
                tick = str(y)
                assert(len(tick) in {1, 2, 3})
                if len(str(y)) == 1 and y % 5 == 0:
                    tick = str(y) + "-|"
                elif len(str(y)) == 1:
                    tick = "  " + "|"
                elif len(str(y)) == 2 and y % 5 == 0:
                    tick = str(y) + "|"
                elif len(str(y)) == 2:
                    tick = "  " + "|"
                # assert(len(tick) == TICK_WIDTH)
                print(tick, end="")
            # print point
            square = "."

            if is_mean:
                square_color = COLORS[square_class]
                square = CLUSTERS[square_class]
                square = f"{square_color}{square}{RESET}"
            elif is_sample:
                square_color = COLORS[square_class]
                square = CLASSES[square_class]
                square = f"{square_color}{square}{RESET}"
            else:
                nearest_mean = min(means, key=lambda mean: distance((x, y), mean))
                square_class = means.index(nearest_mean)

                square_color = COLORS[square_class]
                square_class = CLASSES[square_class]
                square = f"{square_color}{square}{RESET}"
            print(square, end="")

        print()

    assert means_seen_len == len(means), f"means_seen_len = {means_seen_len}, len(means) = {len(means)}"


    # print x ticks
    print(" " * TICK_WIDTH, end="")
    for x in range(X_MIN, X_MAX):
        tick = str(x)
        print(f"=", end="")
    print()

    print(" " * TICK_WIDTH, end="")
    for x in range(X_MIN, X_MAX):
        tick = str(x)
        if x % 5 == 0:
            print(f"|", end="")
        else:
            print(" ", end="")
    print()

    # print x tick digits
    print(" " * TICK_WIDTH, end="")
    for x in range(X_MIN, X_MAX):
        tick = str(x)
        if x % 5 == 0 and len(tick) == 1:
            print(f"{x}", end="")
        else:
            print(" ", end="")
    print()


# TODO: make this a class
# TODO: can I compute the distances all in 1 step?
# TODO: I think I can get this to flip infinitely? What does it remind me of? Game of life? Something else...
# TODO: I want to generate all possible mean centers for this small dataset
# TODO: What metrics can I run to test how well this is performing?
def kmeans_cluster(dataset, means_len):
    sample_xs = [sample_x for (sample_x, sample_y) in dataset]
    sample_ys = [sample_y for (sample_x, sample_y) in dataset]

    min_sample_x = math.floor(min(sample_xs))
    max_sample_x = math.ceil(max(sample_xs))
    min_sample_y = math.floor(min(sample_ys))
    max_sample_y = math.ceil(max(sample_ys))

    means = [None for _ in range(means_len)]
    dataset_mean_assignments = [None for _ in range(len(dataset))]

    # initialization (TODO: just take random points in the dataset)
    for mean_ix in range(means_len):
        mean_x = random.randint(min_sample_x, max_sample_x) # TODO: use my own rng?
        mean_y = random.randint(min_sample_y, max_sample_y)
        means[mean_ix] = [mean_x, mean_y]
    print(f"kmeans: initialized means at {means}")

    steps = 0
    total_dt = 0
    while True:
        tic = time.time_ns()
        # means = [[2, 2], [9, 9]]
        # means = [[1, 6], [1, 8]]
        # [[4.5, 5.5], [5.5, 5.0]]
        assignment_changes_len = 0
        # assignment
        for (sample_ix, sample) in enumerate(dataset):
            nearest_mean = min(means, key=lambda mean: distance(mean, sample))
            nearest_mean_ix = means.index(nearest_mean) # TODO: this is slow
            if dataset_mean_assignments[sample_ix] != nearest_mean_ix:
                assignment_changes_len += 1
                dataset_mean_assignments[sample_ix] = nearest_mean_ix
        if assignment_changes_len == 0:
            break
        steps += 1
        kmeans_dataset_plot(dataset, means, dataset_mean_assignments)
        # update
        for mean_ix in range(means_len):
            assigned_samples = [sample for (sample, sample_mean_ix) in zip(dataset, dataset_mean_assignments) if sample_mean_ix == mean_ix]
            if len(assigned_samples) > 0:
                assigned_sample_xs = [sample_x for (sample_x, _) in assigned_samples]
                assigned_sample_ys = [sample_y for (_, sample_y) in assigned_samples]

                assigned_sample_x_sum = sum(assigned_sample_xs)
                assigned_sample_y_sum = sum(assigned_sample_ys)

                assigned_sample_x_mean = (1 / len(assigned_samples)) * assigned_sample_x_sum
                assigned_sample_y_mean = (1 / len(assigned_samples)) * assigned_sample_y_sum

                assigned_sample_mean = [assigned_sample_x_mean, assigned_sample_y_mean]

                means[mean_ix] = assigned_sample_mean
        print(f"kmeans: means = {means}")
        toc = time.time_ns()
        dt = toc - tic
        total_dt += dt
        print(f"kmeans: dt = {dt}ns")
        print(f"kmeans: throughput = {total_dt // steps}ns/it")
        time.sleep(1)

    print()
    if steps == 1:
        print(f"kmeans: converged to {means} in {steps} step")
    else:
        print(f"kmeans: converged to {means} in {steps} steps")
    print(f"kmeans: dt = {total_dt}ns")
    print(f"kmeans: throughput = {total_dt // steps}ns/it")
    return (means, dataset_mean_assignments)

def main():
    dataset = DATA_SMALL
    print(f"kmeans: len(dataset) = {len(dataset)}")
    dataset_distance_matrix = distance_matrix_from(dataset)
    # dataset_plot(dataset)
    if len(dataset) <= 10:
        distance_matrix_display(dataset_distance_matrix)

    # TODO what is the avg number of iters kmeans runs?
    k = 4
    (means, assignments) = kmeans_cluster(dataset, k)
    # dataset_plot(dataset + means)

    true_clusters = [[2, 2], [9, 9]]
    true_cluster_assignments = [0, 0, 0, 1, 1, 1]

    supervised_dataset = [[x, k] for (x, k) in zip(dataset, true_cluster_assignments)]
    # supervised_dataset_plot(supervised_dataset)

if __name__ == '__main__':
    main()


# TODO:
# [ ] add dataset files
# [ ] mnist with k = 10
# [ ] iris dataset
# [ ] yeast from murphy book
# [ ] diagrams from the bottom of that kaggle notebook
# [ ] mixture of gaussians
# [ ] fireframe 3d plot
