#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# Exercise 2.1: Another ball dropped from a tower
# A ball is again dropped from a tower of height h with initial velocity zero. Write a
# program that asks the user to enter the height in meters of the tower and then calculates
# and prints the time the ball takes until it hits the ground, ignoring air resistance. Use
# your program to calculate the time for a ball dropped from a 100 m high tower.

# By the kinematic equations, h = 0.5gt^2.
# Solving for t yields t = (2g/h)^1/2

g = 9.80665

h = float(input("Enter the height of the tower: "))
t = ((2 * h) / g) ** 0.5

print(f"The ball will hit the ground in {t} seconds")

# TODO: How are the kinematic formulas derived?
ear
