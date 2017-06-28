Sample code for the [hstdg2017 paper][hstdg2017]
================================================

This repository contains sample code for the h-adaptive space-time
Discontinuous Galerkin method described in the paper [*Efficient simulation of
one-dimensional two-phase flow with a high-order h-adaptive space-time
Discontinuous Galerkin method*][hstdg2017].  The code is written in [Python]
and uses the Finite Element library [Nutils].

Requirements
------------

The code requires Python 3.5 or higher — version 3.4 might work as well, but is
not tested — and Nutils 3, which is at the time of writing available as beta on
[github][Nutils-repo].

Examples
--------

You can start an example by executing

    python3 hstdg.py EXAMPLE

where `EXAMPLE` is one of `burgers`, a simple test using Burgers' equation, or
`ifp`, the test case described in the paper.  Run

    python3 hstdg.py --help

for Nutils-specific options and

    python3 hstdg.py EXAMPLE --help

for test-case-specific options.

[hstdg2017]: https://doi.org/10.1016/j.compfluid.2017.06.010
[Python]: https://www.python.org/
[Nutils]: http://nutils.org/
[Nutils-repo]: https://github.com/nutils/nutils
