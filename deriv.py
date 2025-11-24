#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# Exercise 4.3: Calculating derivatives
# Suppose we have a function f(x) and we want to calculate its derivative at a point x.
# We can do that with pencil and paper if we know the mathematical form of the function,
# or we can do it on the computer by making use of the definition of the derivative:
#
#              df/df = lim_{delta -> 0} f(x + delta) - f(x) / delta.
# On the computer we can't actually take the limit as delta goes to zero, but we can get
# a reasonable approximation just by making delta small.
#
# (a) Write a program that defines a function f(x) returning the value x(x - 1), then
#     calculates the derivative of the function at the point x = 1 using the formula
#     above with delta = 10^-2. Calculate the true value of the same derivative analytically
#     and compare with the answer your program gives. The two will not agree perfectly. Why
#     not?
#
# (b) Repeat the calculation for delta = 10^-4, 10^-6, 10^-8, 10^-10, 10^-12, and 10^-14 • You
#     should see that the accuracy of the calculation initially gets better as 6 gets smaller,
#     but then gets worse again. Why is this? TODO

DELTA = 10 ** -2

deltas = [10 ** -4, 10 ** -6, 10 ** -8, 10 ** -12, 10 ** -14]

def f(x):
    return x * (x - 1)

x = float(input("Enter x: "))

deriv = (f(x + DELTA) - f(x)) / DELTA

print(f"The derivative is: {deriv}")


for d in deltas:
    print(f"The derivative for delta={d} is: {(f(x + d) - f(x)) / d}")
