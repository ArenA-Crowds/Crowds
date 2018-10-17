#-------------------------------------------------------------------------------
#
# This code is a stripped down and adapted version of the Scipy
#
# kernel density estimation class (scipy/scipy/stats/kde.py).
#
#-------------------------------------------------------------------------------


import numpy as np

class variable_kde:

    '''This class is customized for bivariate variable kernel density
    estimation, where the estimator bandwidth varies for each data point,
    and is given together with the data points.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. This is a 2-D array with
        shape (2, # of data).

    errors : array_like
        An associated set of uncertainty values. This is a 2-D array with
        shape (2, # of data), the order of which corresponds
        with the data points.

    '''

    # we assume dataset and errors are 2 x n arrays.

    def __init__(self, dataset, errors):
        self.dataset = dataset
        self.errors = errors
        self.n = self.dataset.shape[1]

    # we assume that points is a length m vector representing grid points

    def evaluate(self, points):

        """Evaluate the estimated pdf on a provided set of points.

        Parameters
        ----------
        points : (2, # of points)-array
            This should be the result of unraveling the 2-D evaluation grid.

        Returns
        -------
        values : (# of points,)-array
            The values at each point. It should be reshaped to the size
            of the evaluation grid.

        Method
        ______

        For each data point we compute its contribution to all evaluation
        grid points at once. Because each data point comes with its own
        error values, we create a covariance matrix and bivariate
        normalization factor for each point separately.
        If the error in either the x or y direction is zero, the covariance
        matrix is singular and cannot be inverted. In that case the nearest
        evaluation grid point is set to unity.
        Finally the result is normalized by the number of data points.

        """

        # the number of evaluation points
        self.m = points.shape[1]

        result = np.zeros((self.m,), dtype=float)

        # we loop over the data points
        for i in range(self.n):
            diff = self.dataset[:, i, np.newaxis] - points

            try:
                self._compute_covariance(self.errors[:,i])
                tdiff = np.dot(self.inv_cov, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                result = result + np.exp(-energy) / self._norm_factor
            except np.linalg.LinAlgError:
                # the nearest grid point is set to one (or zero)
                result[np.argmin(np.sum(abs(diff), axis=0))] += 0

        result = result / self.n

        return result

    __call__ = evaluate

    def _compute_covariance(self, sigmas):
        covariance = np.array([[sigmas[0]**2, 0], [0, sigmas[1]**2]])
        self.inv_cov = np.linalg.inv(covariance)
        self._norm_factor = np.sqrt(np.linalg.det(2 * np.pi * covariance))

