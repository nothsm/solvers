#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# Exercise 2.2: Altitude of a satellite
# A satellite is to be launched into a circular orbit around the Earth so that it orbits the
# planet once every T seconds.
#
# (a) Show that the altitude h above the Earth's surface that the satellite must have is
#                           h = ((G M T^2) / (4 pi^2))^1/3 - R
#     where G = 6.67 * 10^-11 m^3 kg^-1 s^-2 is Newton's gravitational constant, M = 5.97 * 10^24 kg
#     is the mass of the Earth, and R = 6371 km is its radius.
#
#     TODO
#
# (b) Write a program that asks the user to enter the desired value of T and then calculates and prints
#     out the correct altitude in meters. TODO: check
#
# (c) Use your program to calculate the altitudes of satellites that orbit the Earth once
#     a day (so-called "geosynchronous" orbit), once every 90 minutes, and once every
#     45 minutes. What do you conclude from the last of these calculations? TODO
#
# (d) Technically a geosynchronous satellite is one that orbits the Earth once per sidereal
#     day, which is 23.93 hours, not 24 hours. Why is this? And how much difference
#     will it make to the altitude of the satellite? TODO

import math

G = 6.67 * (10 ** -11) # newton's gravitational constant
M = 5.97 * (10 ** 24)  # mass of the earth
R = 6371 * 1000        # radius of the earth

DAY = 86400           # 24    hours
SIDEREAL_DAY = 86148  # 23.93 hours
MINS1 = 5400          # 90    minutes
MINS2 = 2700          # 45    minutes

T = float(input("Enter the time interval (in seconds) of the satellite to orbit Earth: "))
h = (((G * M * (T ** 2)) / (4 * (math.pi ** 2))) ** (1 / 3)) - R


print(f"The altitude of the satellite above Earth is {h} meters")
