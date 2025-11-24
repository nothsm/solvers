#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# Exercise 2.3: Write a program to perform the inverse operation to that of Example 2.2.
# That is, ask the user for the Cartesian coordinates x, y of a point in two-dimensional
# space, and calculate and print the corresponding polar coordinates, with the angle 9
# given in degrees.

import math

x = float(input("Enter x: "))
y = float(input("Enter y: "))

r = math.sqrt((x ** 2) + (y ** 2))
d = math.degrees(math.atan2(y, x)) # TODO: check domain/range, and sign

print(f"r = {r}, theta = {d}")
