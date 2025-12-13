import math
import random
from typing import Callable

WIDTH = 1.0 # width of the two pieces of wood
NEEDLE_LEN = 1.0

def monte_carlo(n_trials: int, experiment: Callable[[], bool]) -> float:
    n_passed = sum(int(experiment()) for _ in range(n_trials))
    return n_passed / n_trials

def buffon_test():
    mid, angle = random.uniform(0, WIDTH), random.uniform(0, math.pi)
    dy = min(WIDTH - mid, mid)
    return dy <= (NEEDLE_LEN / 2) * math.sin(angle)

def estimate_pi(n_trials):
    p = monte_carlo(n_trials, buffon_test)
    return (2 * NEEDLE_LEN) / (WIDTH * p)

def main():
    n_trials = int(input("Enter the number of trials to simulate: ").strip())
    print(f"Pi is approximately: {estimate_pi(n_trials)}")

if __name__ == '__main__':
    main()