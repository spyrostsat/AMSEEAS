"""
.. module:: experimental_design
  :synopsis: Methods for generating an experimental design.
.. moduleauthor:: David Eriksson <dme65@cornell.edu>,
                 Yi Shen <ys623@cornell.edu>
:Module: experimental_design
:Author: David Eriksson <dme65@cornell.edu>
        Yi Shen <ys623@cornell.edu>
"""

import numpy as np


class SymmetricLatinHypercube(object):
    """Symmetric Latin Hypercube experimental design
    :param dim: Number of dimensions
    :type dim: int
    :param npts: Number of desired sampling points
    :type npts: int
    :ivar dim: Number of dimensions
    :ivar npts: Number of desired sampling points
    """

    def __init__(self, dim, npts):
        self.dim = dim
        self.npts = npts

    def _slhd(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit hypercube
        :return: Symmetric Latin hypercube design in the unit cube of size npts x dim
        :rtype: numpy.array
        """

        # Generate a one-dimensional array based on sample number
        points = np.zeros([self.npts, self.dim])
        points[:, 0] = np.arange(1, self.npts+1)

        # Get the last index of the row in the top half of the hypercube
        middleind = self.npts//2

        # special manipulation if odd number of rows
        if self.npts % 2 == 1:
            points[middleind, :] = middleind + 1

        # Generate the top half of the hypercube matrix
        for j in range(1, self.dim):
            for i in range(middleind):
                if np.random.random() < 0.5:
                    points[i, j] = self.npts-i
                else:
                    points[i, j] = i + 1
            np.random.shuffle(points[:middleind, j])

        # Generate the bottom half of the hypercube matrix
        for i in range(middleind, self.npts):
            points[i, :] = self.npts + 1 - points[self.npts - 1 - i, :]

        return points/self.npts

    def generate_points(self):
        """Generate a matrix with the initial sample points,
        scaled to the unit hypercube
        :return: Symmetric Latin hypercube design in the unit cube of size npts x dim
            that is of full rank
        :rtype: numpy.array
        :raises ValueError: Unable to find an SLHD of rank at least dim + 1
        """

        rank_pmat = 0
        pmat = np.ones((self.npts, self.dim + 1))
        xsample = None
        max_tries = 100
        counter = 0
        while rank_pmat != self.dim + 1:
            xsample = self._slhd()
            pmat[:, 1:] = xsample
            rank_pmat = np.linalg.matrix_rank(pmat)
            counter += 1
            if counter == max_tries:
                raise ValueError("Unable to find a Symmetric Latin hypercube Design for the desired amount of control variables, please increase m ( >= dim+1 )")
        return xsample