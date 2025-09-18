#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "matplotlib"
# ]
# ///

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np

DEBUG = bool(os.getenv('DEBUG', False))
VERIFY = bool(os.getenv('VERIFY', False))
DEPTH = int(os.getenv('DEPTH', 4))
SPEC = os.getenv('SPEC', 'add1')

SYMBOLS = {'x'}

# f(x) = x + 1
ADD1_DATA = [
    (3, 4),
    (7, 8),
    (1, 2)
]

# f(x) = 2 * x = x + x
MUL2_DATA = [
    (3, 6),
    (7, 14),
    (1, 2)
]

# F(0) = 0, F(1) = 1, F(n) = F(n - 1) + F(n - 2)
FIB_DATA = [
    (0, 0),
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 3)
]

# f(x) = sqrt(x)
# TODO: probably need a tolerance here
SQRT_DATA = [
    (3, math.sqrt(3)),
    (7, math.sqrt(7)),
    (1, math.sqrt(1))
]

def pshow(p):
    if len(p) == 1:
        (c, ) = p
        return c
    elif len(p) == 2:
        (a, op) = p
        return f"{op}({pshow(a)})"
    elif len(p) == 3:
        (a1, a2, op) = p
        return f"{op}({pshow(a1)}, {pshow(a2)})"
    elif len(p) == 4:
        (a1, a2, a3, op) = p
        return f"{op}({pshow(a1)}, {pshow(a2)}, {pshow(a3)})"
    else:
        raise ValueError(f"badsci: pshow: unexpected number of args ({len(p)})")

def pcompile0(p):
    (c, ) = p

    if c in SYMBOLS:
        def f(x):
            return x
    else:
        def f(x):
            return int(c)

    return f

def pcompile1(p):
    (a, op) = p

    g = pcompile(a)

    if op == 'f':
        def f(x, n=10):
            if n == 0:
                return None

            return f(g(x), n=n-1)
    else:
        raise ValueError(f"badsci: pcompile1: unexpected operation ({op})")

    return f

def pcompile2(p):
    (a1, a2, op) = p

    f1 = pcompile(a1)
    f2 = pcompile(a2)

    if op == 'add':
        def f(x):
            return f1(x) + f2(x)
    elif op == 'mul':
        def f(x):
            return f1(x) * f2(x)
    else:
        raise ValueError(f"badsci: pcompile2: unexpected operation ({op})")

    return f

def pcompile(p):
    narg = len(p) - 1

    if narg == 0:
        return pcompile0(p)
    elif narg == 1:
        return pcompile1(p)
    elif narg == 2:
        return pcompile2(p)
    else:
        raise ValueError(f"badsci: pcompile: pcompile doesnt support {narg} args")


def main():
    # dsl (arithmetic/peano)
    consts = ['x', '0', '1', '2', '-1', '-2'] # TODO: restore the rest of the negatives
    # unops = ['f']
    unops = []
    binops = ['add', 'mul']
    # ternops = ['if']
    ternops = []

    programs = [set(), {(c,) for c in consts}]

    depth = 2
    tic = time.perf_counter_ns()
    while depth < DEPTH:
        ps= set()
        frontier = programs[depth - 1]

        # generate all uops
        for p in frontier:
            for op in unops:
                ps.add((p, op))

        # generate all binops
        for p in frontier:
            for q in frontier:
                for op in binops:
                    ps.add((p, q, op))

        # generate all ternops
        for p in frontier:
            for q in frontier:
                for r in frontier:
                    for op in ternops:
                        ps.add((p, q, r, op))

        programs.append(ps)
        depth += 1
    dt = time.perf_counter_ns() - tic

    candidates = []
    if VERIFY:
        spec = ''
        if SPEC == 'add1':
            spec = ADD1_DATA
        elif SPEC == 'mul2':
            spec = MUL2_DATA
        else:
            raise ValueError('badsci: main: unrecognized spec')

        for ps in programs:
            for p in ps:
                f = pcompile(p)

                tic = time.time()
                if all(f(x) == t for (x, t) in spec):
                    candidates.append((p, f))

    if DEBUG:
        for (p, f) in candidates:
            print(pshow(p))

        for ps in programs:
            print()
            for p in ps:
                print(pshow(p))

        print()
        print(f"badsci: dt: {dt}ns")
        print("badsci: candidates:", len(candidates))
        print("badsci: programs:", sum(len(ps) for ps in programs))


if __name__ == '__main__':
    main()
