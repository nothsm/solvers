#!/usr/bin/env -S PYTHON_JIT=1 uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy",
#   "matplotlib"
# ]
# ///

import math
import os
import time
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

COLLECT_LAST = bool(int(os.getenv('COLLECT_LAST', False)))
DEBUG = bool(os.getenv('DEBUG', False))
VERIFY = bool(os.getenv('VERIFY', False))
DEPTH = int(os.getenv('DEPTH', 4))
SPEC = os.getenv('SPEC', 'add1')
PRINT_EVERY = int(os.getenv('PRINT_EVERY', 1_000_000))

SYMBOLS = {'x'}

# CACHE_SIZE = 2048
CACHE_SIZE = 0

RED = "\033[91m"
BLUE = "\033[34m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
WHITE = "\033[37m"
RESET = "\033[0m"

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
    (3, 2),
    (7, 13),
    (1, 1)
]

# f(x) = sqrt(x)
# TODO: probably need a tolerance here
SQRT_DATA = [
    (3, math.sqrt(3)),
    (7, math.sqrt(7)),
    (1, math.sqrt(1))
]

def pshow(p):
    is_operator = isinstance(p, str)

    if is_operator:
        return p
    elif len(p) == 1:
        (c, ) = p
        return c
    elif len(p) == 2:
        (a, op) = p
        return f"{pshow(op)}({pshow(a)})"
    elif len(p) == 3:
        (a1, a2, op) = p
        return f"{pshow(op)}({pshow(a1)}, {pshow(a2)})"
    elif len(p) == 4:
        (a1, a2, a3, op) = p
        return f"{pshow(op)}({pshow(a1)}, {pshow(a2)}, {pshow(a3)})"
    elif len(p) == 5:
        (a1, a2, a3, a4, op) = p
        return f"{pshow(op)}({pshow(a1)}, {pshow(a2)}, {pshow(a3)}, {pshow(a4)})"
    elif len(p) == 6:
        (a1, a2, a3, a4, a5, op) = p
        return f"{pshow(op)}({pshow(a1)}, {pshow(a2)}, {pshow(a3)}, {pshow(a4)}, {pshow(a5)})"
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
    elif op == 'le':
        def f(x):
            return f1(x) <= f2(x)
    else:
        raise ValueError(f"badsci: pcompile2: unexpected operation ({op})")

    return f

def pcompile5(p):
    (a1, a2, binop, a3, a4, op) = p

    test = pcompile(a1)
    base = pcompile(a2)
    decl = pcompile(a3)
    decr = pcompile(a4)

    if op == 'treerec':
        if binop == 'add':
            def f(x, n=10):
                if n == 0 or test(x):
                    return base(x)
                else:
                    return f(decl(x), n=n-1) + f(decr(x), n=n-1)

        elif binop == 'mul':
            def f(x, n=10):
                if n == 0 or test(x):
                    return base(x)
                else:
                    return f(decl(x), n=n-1) * f(decr(x), n=n-1)
        elif binop == 'le':
            def f(x, n=10):
                if n == 0 or test(x):
                    return base(x)
                else:
                    return f(decl(x), n=n-1) <= f(decr(x), n=n-1)
        else:
            raise ValueError(f"badsci: pcompile5: unexpected binary op ({binop})")
    else:
        raise ValueError(f"badsci: pcompile5: unexpected operation ({op})")

    return f

def pcompile(p):
    narg = len(p) - 1

    if narg == 0:
        return pcompile0(p)
    elif narg == 1:
        return pcompile1(p)
    elif narg == 2:
        return pcompile2(p)
    elif narg == 5:
        return pcompile5(p)
    else:
        raise ValueError(f"badsci: pcompile: pcompile doesnt support {narg} args")

def peval():
    ...

def penumerate(dsl):
    programs = [set(), {(c,) for c in dsl[0]}]
    depth = 2
    while depth < DEPTH:
        ps = set()
        frontier = [p for ps in programs for p in ps]

        # generate all uops
        for op in dsl[1]:
            for p in frontier:
                prog = (p, op)
                ps.add(prog)

        # generate all binops
        for op in dsl[2]:
            for p in frontier:
                for q in frontier:
                    prog = (p, q, op)
                    ps.add(prog)

        # generate all ternops
        for op in dsl[3]:
            for p in frontier:
                for q in frontier:
                    for r in frontier:
                        prog = (p, q, r, op)
                        ps.add(prog)

        # generate all quartops
        for op in dsl[4]:
            for p in frontier:
                for q in frontier:
                    for r in frontier:
                        for s in frontier:
                            prog = (p, q, r, s, op)
                            ps.add(prog)

        # generate all quinops
        for op in dsl[5]:
            for p in frontier:
                for q in frontier:
                    for binop in dsl[2]:
                        for s in frontier:
                            for t in frontier:
                                prog = (p, q, binop, s, t, op)
                                ps.add(prog)

        programs.append(ps)
        depth += 1
    return programs

def pbacktrack():
    ...

def pevo():
    ...

def panneal():
    ...

def pmc():
    ...

def main():
    # dsl (arithmetic/peano)

    consts = ['x', '0', '1', '2', '-1', '-2'] # TODO: restore the rest of the negatives
    # unops = ['f']
    unops = []
    binops = ['add', 'mul', 'le']
    # ternops = ['if']
    # ternops = ['add3']
    ternops = []
    # quatops = ['rec']
    # quatops = []
    quinops = ['treerec']

    dsl = {
        0: consts,
        1: unops,
        2: binops,
        3: ternops,
        4: [],
        5: quinops
    }

    spec = ''
    if SPEC == 'add1':
        spec = ADD1_DATA
    elif SPEC == 'mul2':
        spec = MUL2_DATA
    elif SPEC == 'fib':
        spec = FIB_DATA
    else:
        raise ValueError(f'badsci: main: unrecognized spec ({SPEC})')

    programs = [set(), {(c,) for c in dsl[0]}]
    candidates = []

    tic = time.perf_counter_ns()

    depth = 2
    i = 0
    while depth < DEPTH:
        ps = set()
        frontier = [p for ps in programs for p in ps]

        # generate all uops
        for op in dsl[1]:
            for p in frontier:
                prog = (p, op)

                i += 1
                if PRINT_EVERY > 0 and i % PRINT_EVERY == 0:
                    print(i, pshow(prog))

                if depth < DEPTH - 1 or (depth == DEPTH - 1 and COLLECT_LAST):
                    ps.add(prog)

        # generate all binops
        for op in dsl[2]:
            for p in frontier:
                for q in frontier:
                    prog = (p, q, op)

                    i += 1
                    if PRINT_EVERY > 0 and i % PRINT_EVERY == 0:
                        print(i, pshow(prog))

                    if depth < DEPTH - 1 or (depth == DEPTH - 1 and COLLECT_LAST):
                        ps.add(prog)

        # generate all ternops
        for op in dsl[3]:
            for p in frontier:
                for q in frontier:
                    for r in frontier:
                        prog = (p, q, r, op)

                        i += 1
                        if PRINT_EVERY > 0 and i % PRINT_EVERY == 0:
                            print(i, pshow(prog))

                        if depth < DEPTH - 1 or (depth == DEPTH - 1 and COLLECT_LAST):
                            ps.add(prog)

        # generate all quartops
        for op in dsl[4]:
            for p in frontier:
                for q in frontier:
                    for r in frontier:
                        for s in frontier:
                            prog = (p, q, r, s, op)

                            i += 1
                            if PRINT_EVERY > 0 and i % PRINT_EVERY == 0:
                                print(i, pshow(prog))

                            if depth < DEPTH - 1 or (depth == DEPTH - 1 and COLLECT_LAST):
                                ps.add(prog)

        # generate all quinops
        if depth == DEPTH - 1:
            for op in dsl[5]:
                for p in frontier:
                    for q in frontier:
                        for binop in binops:
                            for s in frontier:
                                for t in frontier:
                                    prog = (p, q, binop, s, t, op)

                                    i += 1
                                    if PRINT_EVERY > 0 and i % PRINT_EVERY == 0:
                                        print(i, pshow(prog))

                                    f = pcompile(prog)
                                    if all(f(x) == t for (x, t) in FIB_DATA):
                                        print(f"{GREEN}{pshow(prog)}{RESET}")
                                        candidates.append((prog, f))

                                    if depth < DEPTH - 1 or (depth == DEPTH - 1 and COLLECT_LAST):
                                        ps.add(prog)

        programs.append(ps)
        depth += 1
    dt = time.perf_counter_ns() - tic

    if VERIFY:
        for ps in programs:
            for p in ps:
                f = pcompile(p)

                tic = time.time()
                if all(f(x) == t for (x, t) in spec):
                    candidates.append((p, f))

    if DEBUG:
        for (p, f) in candidates:
            print(f"{GREEN}{pshow(p)}{RESET}")

        for ps in programs:
            print()
            for p in ps:
                print(pshow(p))

        print()
        print(f"badsci: dt: {dt}ns")
        print("badsci: programs:", sum(len(ps) for ps in programs))
        print("badsci: candidates:", len(candidates))


if __name__ == '__main__':
    main()


"""
f(0) = 1
f(1) = 1
f(2) = 1
f(x) = f(x - 1) + f(x - 2)

if x == 0:
    return 0
else:
    if x == 1:
        return 1
    else:
        return f(x - 1) + f(x - 2)

if(eq(x, 0),
   0
   if(eq(x, 1),
      1,
      add(f(add(x, -1)),
          f(add(x, -2)))))

---

if x == 0:
    return 0
else:
    return 1 + f(x - 1)

if(eq(x, 0), 0, add(1, f(add(x, -1))))

# eq(x, 0)
def f1(x):
  return x == 0

# 0
def f2(x):
  return 0

# 1
def f3_1(x):
  return 1

# x
def f3_2_1_1(x):
  return x

# -1
def f3_2_1_2(x):
  return -1

# add(x, -1)
def f3_2_1(x):
  return f3_2_1_1(x) + f3_2_1_2(x)

# f(add(x, -1))
def f3_2(x, op):
  return op(f3_2_1(x))

# add(1, f(add(x, -1)))
def f3(x):
  return f3_1(x) + f3_2(x)

def f(x):
  if f1(x):
    return f2(x)
  else:
    return f3(x)
---

rec(test, base, combiner, dec)
def f(x):
    if test(x):
        return base(x)
    else:
        return combiner(x, f(dec(x)))

combiner is a binop!
rec(test, base, combiner, dec)
def f(x):
  if test(x):
    return base(x)
  else:
    return combiner(x, f(dec(x)))

treerec(test, base, combiner, decl, decr)
def f(x):
  if test(x):
    return base(x)
  else:
    return combiner(x, f(decl(x)), f(decr(x)))

rec2(eq(x, 0), 0, eq(x, 1), 1, add, add(x, -1))


---

I should probably cache generated operators

# const(x)
def f_cx(x):
  return x

# const(2)
def f_c2(x):
  return 2

# const(1)
f _c1(x):
  return 1

#


treerec(le(const(x), const(2)),
        const(1),
        add(const(x), const(y)),
        add(const(x), const(-1)),
        add(const(x), const(-2)))

---

if(le(x, 1),
   1
   add(f(add(x, -1)),
       f(add(x, -2))))

---

if(0, 1, f(add(x, -1)))
"""
