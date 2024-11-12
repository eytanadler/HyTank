# HyTank

[![Build](https://github.com/eytanadler/HyTank/actions/workflows/hytank.yaml/badge.svg?branch=main)](https://github.com/eytanadler/HyTank/actions/workflows/hytank.yaml)

HyTank is a toolkit for modeling the behavior of cryogenic liquid hydrogen tanks.
It includes models for boil-off (including fuel extraction/filling and heater operation), heat leak, and tank weight.

For more details, see the [journal paper](https://www.researchgate.net/publication/385741253).

<h2 align="center">
    <img src=".github/tank_diagram.png" width="500" />
</h2>

## Installation

Make sure that you have [Python](https://www.python.org/) and [pip](https://pip.pypa.io/en/stable/) installed on your computer.
Then install HyTank:

1. Clone this repository to your computer.
2. Open a terminal and navigate the root directory of the HyTank repo you just cloned (the one with the setup.py file in it).
3. Into the terminal type `pip install .`. Alternatively, install it in place with `pip install -e .` if you plan to edit HyTank's files and want the changes to be reflected when you run it.
4. If you want to run the examples and be able to generate plots, install the plotting dependencies with `pip install .[plot]`. In some terminals, you may need to escape the square brackets by instead using `pip install .\[plot\]`.
5. If you want to test your installation, install the testing dependencies with `pip install .[test]`. You may need to escape the backslashes similarly to in the previous step.

### Testing your installation

Once you've installed HyTank, you can ensure that it works by running the test suite.
First, make sure you've installed the testing dependencies (see step 5 above).
Then run the tests by entering the root directory of the package and running the following command:
```
pytest -v .
```
This should print out the list of tests as it runs them and tell you whether they pass.

## Getting started

Have a look in the examples folder, which contains sample scripts that run the model and visualize the results.

## Citation

Please cite this software by reference to the [journal paper](https://www.researchgate.net/publication/385741253):

Eytan J. Adler and Joaquim R. R. A. Martins, "Liquid hydrogen tank boil-off model for design and optimization", Journal of Thermophysics and Heat Transfer, November 2024.

```
@article{Adler2024b,
	author = {Eytan J. Adler and Joaquim R. R. A. Martins},
	issn = {1533-6808},
	journal = {Journal of Thermophysics and Heat Transfer},
	month = {November},
	publisher = {American Institute of Aeronautics and Astronautics},
	title = {Liquid hydrogen tank boil-off model for design and optimization},
	year = {2024}
}
```
