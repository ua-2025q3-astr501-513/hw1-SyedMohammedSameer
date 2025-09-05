#!/usr/bin/env python3
#
# Please look for "TODO" in the comments, which indicate where you
# need to write your code.
#
# Part 4: Solve the Coupled Simple Harmonic Oscillator Problem (1 point)
#
# * Objective:
#   Take the coupled harmonic oscillator problem we solved in class
#   and rewrite it using a well-structured Python class.
# * Details:
#   The description of the problem and the solution template can be
#   found in `hw1/p4.py`.
#
# From lecture `02w`, we solve systems of coupled harmonic oscillators
# semi-analytically by numerically solving eigenvalue problems.
# However, the code structure was not very clean, making the code hard
# to reuse.
# Although numerical analysis in general does not require
# object-oriented programming, it is sometime useful to package
# stateful caluation into classes.
# For this assignment, we will provide a template class.
# Your responsibility to implement the methods in the class.


import numpy as np


class CoupledOscillators:
    """A class to model a system of coupled harmonic oscillators.

    Attributes:
        Omega (np.ndarray): array of angular frequencies of the normal modes.
        V     (np.ndarray): matrix of eigenvectors representing normal modes.
        M0    (np.ndarray): initial amplitudes of the normal modes.

    """

    def __init__(self, X0=[-0.5, 0, 0.5], m=1.0, k=1.0):
        """Initialize the coupled harmonic oscillator system.

        Args:
            X0 (list or np.ndarray): initial displacements of the oscillators.
            m  (float):              mass of each oscillator (assumed identical for all oscillators).
            k  (float):              spring constant (assumed identical for all springs).

        """
        # TODO: Construct the stiffness matrix K
        # TODO: Solve the eigenvalue problem for K to find normal modes
        # TODO: Store angular frequencies and eigenvectors
        # TODO: Compute initial modal amplitudes M0 (normal mode decomposition)
        self.X0 = np.array(X0, dtype=float)

        n = len(X0)

        # Step 1: Construct stiffness matrix K
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 2 * k / m  # main diagonal
            if i > 0:
                K[i, i - 1] = -k / m
            if i < n - 1:
                K[i, i + 1] = -k / m

        # Step 2: Solve eigenvalue problem
        eigvals, eigvecs = np.linalg.eigh(K)  # symmetric matrix → eigh

        # Step 3: Store angular frequencies and eigenvectors
        self.Omega = np.sqrt(np.maximum(eigvals, 0))  # avoid negative rounding
        self.V = eigvecs

        # Step 4: Compute initial modal amplitudes (projection of X0)
        self.M0 = self.V.T @ self.X0

    def __call__(self, t):
        """Calculate the displacements of the oscillators at time t.

        Args:
            t (float): time at which to compute the displacements.

        Returns:
            np.ndarray: displacements of the oscillators at time t.

        """
        # TODO: Reconstruct the displacements from normal modes
        M_t = self.M0 * np.cos(self.Omega * t)
        X_t = self.V @ M_t
        return X_t


if __name__ == "__main__":

    # Initialize the coupled oscillator system with default parameters
    co = CoupledOscillators()

    # Print displacements of the oscillators at each time step
    print("Time(s)  Displacements")
    print("----------------------")
    for t in np.linspace(0, 10, num=101):
        X = co(t)             # compute displacements at time t
        print(f"{t:.2f}", X)  # print values for reference
