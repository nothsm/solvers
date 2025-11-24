#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# Exercise 4.2: Quadratic equations
# Consider a quadratic equation ax^2 +bx + c = 0 that has real solutions.
#
# (a) Write a program that takes as input the three numbers, a, b, c, and prints
#     out the two solutions using teh standard formula
#
#                   a = -b +- sqrt(b^2 - 4ac)  / 2a.
#
#     Use your program to compute the solutions of 0.001x^2 + 1000x + 0.001 = 0.
#
# (b) There is another way to write the solutions to a quadratic equation. Multiplying
#     the top and bottom solution above by -b +- sqrt(b^2 - 4ac), show that the solutions
#     can also be written as
#
#                    x = 2c / -b +- sqrt(b^2 - 4ac)
#
#     Add further lines to your program to print these values in addition to the earlier
#     ones and again use the program to solve 0.001x^2 + 1000x + 0.001 = 0. What do you
#     see? How do you explain it? TODO (no proof)
#
# (c) Using what you have learned, write a new program that calculates both roots of a
#     quadratic equation accurately in all cases.

import math

a = float(input("Enter a: "))
b = float(input("Enter b: "))
c = float(input("Enter c: "))

# x1 = (-b + math.sqrt((b ** 2) - 4 * a * c)) / (2 * a) These solutions are off
# x2 = (-b - math.sqrt((b ** 2) - 4 * a * c)) / (2 * a)

t1 = (2 * c) / (-b + math.sqrt((b ** 2) - 4 * a * c))
t2 = (2 * c) / (-b - math.sqrt((b ** 2) - 4 * a * c))

# print(f"The solutions are: {x1}, {x2}")
print(f"The solutions are: {t1}, {t2}")
