#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "scipy",
#   "sympy"
# ]
# ///

import math
import os
import sys
import time
import operator as op

# import matplotlib.pyplot as plt
# import numpy as np

DEBUG = bool(os.getenv('DEBUG', False))
VERIFY = bool(os.getenv('VERIFY', False))
DEPTH = int(os.getenv('DEPTH', 4))
SPEC = os.getenv('SPEC', 'add1')
PRINT_EVERY = int(os.getenv('PRINT_EVERY', 1_000_000))

CONSTANTS = ['x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9']
UNOPS = ['recip', 'sqrt', 'cat']
BINOPS = ['add', 'mul', 'eq', 'le']
TERNOPS = ['if']

SYMBOLS = {'x', 'y', 'z'}

OPS = {
    'recip': lambda x: 1 / x,
    'sqrt': math.sqrt,
    'cat': lambda x: (x, x),
    'add': op.add,
    'mul': op.mul,
    'eq': op.eq,
    'le': op.le
}

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
    # elif len(p) == 4:
    #     (a1, a2, a3, op) = p
    #     return f"{pshow(op)}({pshow(a1)}, {pshow(a2)}, {pshow(a3)})"
    # elif len(p) == 5:
    #     (a1, a2, a3, a4, op) = p
    #     return f"{pshow(op)}({pshow(a1)}, {pshow(a2)}, {pshow(a3)}, {pshow(a4)})"
    # elif len(p) == 6:
    #     (a1, a2, a3, a4, a5, op) = p
    #     return f"{pshow(op)}({pshow(a1)}, {pshow(a2)}, {pshow(a3)}, {pshow(a4)}, {pshow(a5)})"
    else:
        raise ValueError(f"nth: pshow: unexpected number of args ({len(p)})")

# TODO
def pparse(s):
    ...

def pfreevars(p):
    freevars = set()
    def go(p):
        is_constant = isinstance(p, str)

        if is_constant:
            if p in SYMBOLS:
                freevars.add(p)
        else:
            assert isinstance(p, tuple)

            # constants have no operation
            if len(p) == 1:
                (c, ) = p
                go(c)
            else:
                for arg in p[:-1]: # drop the last operation
                    go(arg)
    go(p)
    return freevars

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

    # if op == 'recip':
    #     def f(x):
    #         return 1 / g(x)
    # elif op == 'sqrt':
    #     def f(x):
    #         return g(x) ** 0.5
    # elif op == 'cat':
    #     def f(x):
    #         v = g(x)
    #         return (v, v)
    # else:
    #     raise ValueError(f"nth: pcompile1: unexpected operation ({op})")

    o = OPS[op]

    def f(x):
        return o(g(x))

    return f

def pcompile2(p):
    (a1, a2, op) = p

    f1 = pcompile(a1)
    f2 = pcompile(a2)

    nfree = len(pfreevars(p))

    if nfree <= 1:
        if op == 'mul':
            def f(x):
                return f1(x) * f2(x)
        else:
            raise ValueError(f"nth: pcompile2: unexpected operation ({op})")
    elif nfree == 2:
        if len(pfreevars(a1)) == 1 and len(pfreevars(a2)) == 1:
            if op == 'mul':
                def f(x):
                    (x, y) = x
                    return f1(x) * f2(y)
            else:
                raise ValueError(f"nth: pcompile2: unexpected operation ({op})")
        elif len(pfreevars(a1)) == 2 and len(pfreevars(a2)) == 0:
            if op == 'mul':
                def f(x1, x2):
                    return f1(x1, x2) * f2((x1, x2))
        elif len(pfreevars(a1)) == 0 and len(pfreevars(a2)) == 2:
            if op == 'mul':
                def f(x1, x2):
                    return f1((x1, x2)) * f2(x1, x2)
            else:
                raise ValueError(f"nth: pcompile2: unexpected operation ({op})")
        else:
            raise ValueError(f"nth: pcompile2: unexpected combiunation of frevars ({len(pfreevars(a1)), len(pfreevars(a2))})")
    else:
        raise ValueError(f"nth: pcompile2: unexpected number of free vars ({nfree}), {pshow(p)}")

    return f

    # if op == 'add':
    #     def f(x):
    #         return f1(x) + f2(x)
    # elif op == 'mul':
    #     def f(x):
    #         return f1(x) * f2(x)
    # elif op == 'le':
    #     def f(x):
    #         return f1(x) <= f2(x)
    # else:
    #     raise ValueError(f"nth: pcompile2: unexpected operation ({op})")

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
            raise ValueError(f"nth: pcompile5: unexpected binary op ({binop})")
    else:
        raise ValueError(f"nth: pcompile5: unexpected operation ({op})")

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
        raise ValueError(f"nth: pcompile: pcompile doesnt support {narg} args")

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
                                print(prog)
                                assert False
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
    # dsl: (arithmetic/peano)
    consts = ['x', 'y', 'z', '0', '1', '2', '-1', '-2'] # TODO: restore the rest of the negatives
    # unops = ['f']
    # unops = ['recip', 'sqrt', 'cat']
    unops = ['sqrt']
    # binops = ['add', 'mul', 'le']
    binops = ['mul']
    # binops = []
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
        # 3: ternops,
        3: [],
        4: [],
        # 5: quinops
        5: []
    }

    spec = ''
    if SPEC == 'add1':
        spec = ADD1_DATA
    elif SPEC == 'mul2':
        spec = MUL2_DATA
    elif SPEC == 'fib':
        spec = FIB_DATA
    else:
        raise ValueError(f'nth: main: unrecognized spec ({SPEC})')

    tic = time.perf_counter_ns()
    programs = penumerate(dsl)
    dt = time.perf_counter_ns() - tic

    candidates = []
    errs = set()
    if VERIFY:
        for ps in programs:
            for p in ps:
                try:
                    f = pcompile(p)
                except:
                    print(f"BADCOMPILE!!! {pshow(p)}", file=sys.stderr)
                    continue
                for (x, t) in spec:
                    try:
                        if f(x) != t:
                            break
                    except:
                        print(f"ERROR!!! ({x}, {pshow(p)})", file=sys.stderr)
                        errs.add(p)
                        break

    if DEBUG:
        for (p, f) in candidates:
            print(f"{GREEN}{pshow(p)}{RESET}")

        for ps in programs:
            print()
            for p in ps:
                f = pcompile(p)
                print(pshow(p), p, pfreevars(p), [(x, f(x), t) for (x, t) in spec] if p not in errs else [])

        print()
        print(f"nth: dt: {dt}ns")
        print("nth: programs:", sum(len(ps) for ps in programs))
        print("nth: candidates:", len(candidates))

if __name__ == '__main__':
    main()

# TODO: compile mul(cat(3))

# TODO:
# [ ] Get nary functions to work
# [ ] Get recursive functions to work
# [ ] Get list ops to work
# [ ] Setup infrastructure for using ml to guide search
# [ ] Setup infrastructure for using rl to guide search
# [ ] Setup infrastructure for training ml
# [ ] Setup infrastructure for training rl
# [ ] Get library learning to work
# [ ] High performance data structures
# [ ] Perhaps try version space algebras? Or lattices?

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

"""
problems:
- ARC
- OEIS
- superoptimization
- fib
- bad science
- qsort
- dreamcoder
  - list processing
  - text editing
  - regexes
  - LOGO graphics
  - block towers
  - symbolic regression
  - recursive programming
  - physical laws
- probably used ebm for efficient search

- should try out rosette, smt solvers
"""
