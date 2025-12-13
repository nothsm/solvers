# buffon

This is a little project I wrote for fun that implements the world's worst way of approximating $\pi$.

More specifically, this project runs a Monte Carlo simulation to approximate $\pi$ from the Buffon's Needle Problem (link: https://en.wikipedia.org/wiki/Buffon%27s_needle_problem).

First, program runs a Monte
Carlo simulation that computes
$
p = \frac{n_{pass}}{n_{trial}},
$
where $n_{pass}$ is the number of times dropping the needle crosses one
of the strips of wood. Then, the program approximates $\pi$ using the formula
$
\pi \approx \frac{2l}{tp},
$
where $l$ is the length of the needle, and $t$ is the width between the two strips of wood. The derivation of this equation is not that hard (check the Wikipedia link), but I don't want to bloat this readme. 

You can try tinkering with the hyperparameters in `buffon.py`.

## quick start 
All you have to do is run `python buffon.py` and then enter the number of trials you want. In practice, any number
than more than about 100,000,000 on my M3 Pro 18GB Macbook Pro takes way too long to run.

Alternatively, you can tinker with the experiment in `buffon.ipynb`.