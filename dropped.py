#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

g = 9.80665

h = float(input("Enter the height of the tower: "))
t = float(input("Enter the time interval: "))
s = 0.5 * g * (t ** 2)

print(f"The height of the ball is {h - s} meters")
