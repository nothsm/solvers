#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

import math

r = float(input("Enter r: "))
d = float(input("Enter theta in degree: "))

theta = d * math.pi / 180
x = r * math.cos(theta)
y = r * math.sin(theta)

print(f"x = {x}, y = {y}")
