"""
Bingham Distribution

This module implements the Bingham distribution as it was proposed in:
Christopher Bingham, *"An Antipodally Symmetric Distribution on the Sphere"*,
Annals of Statistics 2(6), 1974
"""
import logging
import scipy.integrate as integrate
import scipy.optimize
import scipy.special

import numpy as np
import sys


class BinghamDistribution(object):
    """Implementation of the Bingham Distribution.

    We represent the Bingham distribution as

    .. math::
         f(x) = \\exp\\left( x^\\top M Z M^\\top x \\right)\\ , \\quad x\\in S^n

    The current implementation supports the 2d and 4d case of the Bingham
    Distribution (i.e. n=2 and n=4).

    Parameters
    ----------
        param_m : array of shape (dim,dim)
            Location and noise direction parameter matrix M of the Bingham
            distribution.

        param_z : array of shape (dim)
            Diagonal entries of dispersion parameter matrix Z of the Bingham
            distribution.

        options : dict
            Dictionary containing additional options that may be:
            "norm_const_mode":
                Mode of computing the normalization constant as described for
                the mode parameter of the normalization_constant method.
            "norm_const_options":
                Optional normalization constant computation options for the
                normalization_constant method. Only processed if norm_const_mode
                is provided.
    """

    # Constant indicating which dimensions are implemented.
    IMPLEMENTED_DIMENSIONS = [2, 4]

    def __init__(self, param_m, param_z, options=dict()):
        self.assert_parameters(param_m, param_z)

        self._dim = param_m.shape[0]
        self._param_m = np.copy(param_m.astype(float))
        self._param_z = np.copy(param_z.astype(float))

        self._mode = self._param_m[:, -1]

        if "norm_const_mode" in options.keys():
            nc_options = options["norm_const_options"] \
                if "norm_const_options" in options.keys() else dict()

            self._norm_const = self.normalization_constant(
                param_z, mode=options["norm_const_mode"], options=nc_options)
        else:
            self._norm_const = self.normalization_constant(param_z)

        self._norm_const_deriv \
            = BinghamDistribution.normalization_constant_deriv(self._param_z)

        self._logger = logging.getLogger(__name__)

    ##############
    # Properties #
    ##############
    @property
    def dim(self):
        return self._dim

    @property
    def m(self):
        return self._param_m

    @property
    def mode(self):
        return self._mode

    @property
    def norm_const(self):
        return self._norm_const

    @property
    def norm_const_deriv(self):
        return self._norm_const_deriv

    @property
    def z(self):
        return self._param_z

    ##################
    # Public Methods #
    ##################
    def is_almost_equal(self, other):
        return (np.allclose(self._param_m, other.m) and
                np.allclose(self._param_z, other.z))

    def pdf(self, data):
        """PDF of the Bingham Distribution

        Parameters
        ----------
        data : array of shape(n_points, dim)
            The samples at which the density is evaluated.

        Returns
        -------
        density : array of shape (n_points),
            Value of the pdf evaluated at each data point.
        """

        assert isinstance(data, np.ndarray), \
            "Samples need bo be of type numpy.ndarray."

        if len(data.shape) == 1:
            data = np.array([data])
        assert len(data.shape) == 2 and data.shape[1] == self._dim, \
            "Sample dimension does not agree with distribution dimension."

        # Here, the Bingham distribution parametrization we use is
        # f(x) \propto exp(x^T M Z M^T x)
        full_param_matrix = \
            np.dot(self._param_m, np.dot(np.diag(self._param_z),
                                         self._param_m.transpose()))

        # This can be later vectorized for speed-up
        num_data_points = np.shape(data)[0]
        density = np.zeros(num_data_points)
        for i in range(0, num_data_points):
            density[i] = np.exp(
                np.dot(data[i], np.dot(full_param_matrix, data[i])))
        density = density / self._norm_const
        return density

    def random_samples(self, n):
        """Generates Bingham random samples.

        The random sampling uses a rejection method that was originally
        proposed in
        J. T. Kent, A. M. Ganeiber, K. V. Mardia, "A New Method to Simulate
        the Bingham and Related Distributions in Directional Data Analysis
        with Applications", arXiv preprint arXiv:1310.8110, 2013.

        Parameters
        ----------
        n : integer
            Number of random samples.

        Returns
        -------
        samples : array of shape (n_points, dim)
            Array with random samples.
        """

        samples = np.zeros([n, self._dim])

        a = -np.dot(
            self._param_m, np.dot(np.diag(self._param_z), self._param_m.T))

        b = scipy.optimize.fsolve(
            lambda x: np.sum(1. / (x - 2. * self._param_z)) - 1,
            1.0
        )[0]
        self._logger.debug("b=%g", b)

        omega = np.eye(self._dim) + 2. * a / b
        mbstar = np.exp(-(self._dim - b) / 2.) \
            * (self._dim / b)**(self._dim / 2.)

        def fb_likelihood(x):
            return np.exp(np.dot(-x, np.dot(a, x.T)))

        def acg_likelihood(x):
            return np.dot(x, np.dot(omega, x.T)) ** (-self._dim / 2.)

        current_sample = 0
        while current_sample < n:
            candidate = np.random.multivariate_normal(
                np.zeros(self._dim), np.linalg.inv(omega), 1)
            candidate = candidate / np.linalg.norm(candidate)

            w = np.random.uniform()
            if w < fb_likelihood(candidate) / (mbstar *
                                               acg_likelihood(candidate)):
                samples[current_sample] = candidate
                current_sample += 1

        return samples

    def second_moment(self):
        """Calculate covariance matrix of Bingham distribution.

        Returns
        -------
        s (d x d matrix): scatter/covariance matrix in R^d
        """
        nc_deriv_ratio = np.diag(self._norm_const_deriv / self._norm_const)

        # The diagonal of D is always 1, however this may not be the
        # case because dF and F are calculated using approximations
        nc_deriv_ratio = nc_deriv_ratio / sum(np.diag(nc_deriv_ratio))

        s = np.dot(self._param_m,
                   np.dot(nc_deriv_ratio, self._param_m.transpose()))
        s = (s + s.transpose()) / 2  # enforce symmetry

        return s

    ##################
    # Static Methods #
    ##################
    @staticmethod
    def multiply(b1, b2):
        """Computes the product of two Bingham pdfs

        This method makes use of the fact that the Bingham distribution
        is closed under Bayesian inference. Thus, the product of two
        Bingham pdfs is itself the pdf of a Bingham distribution. This
        method computes the parameters of the resulting distribution.

        Parameters
        ----------
        b1 : BinghamDistribution
            First Bingham Distribution.
        b2 : BinghamDistribution
            Second Bingham Distribution.

        Returns
        -------
        B : BinghamDistribution
            Bingham distribution representing this*B2 (after
            renormalization).
        """
        assert isinstance(b2, BinghamDistribution), \
            "Second argument needs to be of type BinghamDistribution"

        assert b1.dim == b2.dim, \
            "Dimensions do not match"

        # new exponent
        c = np.add(np.dot(b1.m,
                          np.dot(np.diag(b1.z),
                                 b1.m.transpose())),
                   np.dot(b2.m, np.dot(np.diag(b2.z),
                                       b2.m.transpose())))

        # Ensure symmetry of c, asymmetry may arise as a consequence of a
        # numerical instability earlier.
        c = 0.5 * np.add(c, c.transpose())

        eigvalues, eigvectors = np.linalg.eig(c)  # Eigenvalue decomposition
        eigvalues = eigvalues[::-1]
        indx = eigvalues.argsort()
        z_param = eigvalues[indx]
        # Rows and columns are swapped in numpy's eig
        m_param = eigvectors.transpose()[indx]
        z_param = z_param - z_param[-1]  # last entry should be zero
        return BinghamDistribution(m_param, z_param)

    @staticmethod
    def compose(b1, b2):
        """Compose two Bingham distributions.
        Using Moment Matching based approximation, we compose two Bingham
        distributions. The mode of the new distribution should be the
        quaternion multiplication of the original modes; the uncertainty
        should be larger than before

        Parameters
        ----------
        b1 : BinghamDistribution
            First Bingham Distribution.
        b2 : BinghamDistribution
            Second Bingham Distribution.

        Returns
        -------
        b : BinghamDistribution
            Bingham distribution representing the convolution
        """

        assert b1.dim == b2.dim, \
            "Dimensions not equal"
        assert b1.dim == 2 or b1.dim == 4, \
            "Unsupported dimension"

        b1s = b1.second_moment()
        b2s = b2.second_moment()

        if b1.dim == 2:
            # for complex numbers
            # derived from complex multiplication
            # Gerhard Kurz, Igor Gilitschenski, Simon Julier, Uwe D. Hanebeck,
            # Recursive Bingham Filter for Directional Estimation Involving 180
            # Degree Symmetry Journal of Advances in Information Fusion,
            # 9(2):90 - 105, December 2014.
            a11 = b1s[0, 0]
            a12 = b1s[0, 1]
            a22 = b1s[1, 1]
            b11 = b2s[0, 0]
            b12 = b2s[0, 1]
            b22 = b2s[1, 1]

            s11 = a11 * b11 - 2 * a12 * b12 + a22 * b22
            s12 = a11 * b12 - a22 * b12 - a12 * b22 + a12 * b11
            s21 = s12
            s22 = a11 * b22 + 2 * a12 * b12 + a22 * b11

            s = np.array([[s11, s12], [s21, s22]])

            return BinghamDistribution.fit_to_moment(s)
        else:
            # adapted from Glover's C code in libBingham, see also
            # Glover, J. & Kaelbling, L.P. Tracking 3 - D Rotations with
            # the Quaternion Bingham Filter MIT, 2013

            a11 = b1s[0, 0]
            a12 = b1s[0, 1]
            a13 = b1s[0, 2]
            a14 = b1s[0, 3]
            a22 = b1s[1, 1]
            a23 = b1s[1, 2]
            a24 = b1s[1, 3]
            a33 = b1s[2, 2]
            a34 = b1s[2, 3]
            a44 = b1s[3, 3]

            b11 = b2s[0, 0]
            b12 = b2s[0, 1]
            b13 = b2s[0, 2]
            b14 = b2s[0, 3]
            b22 = b2s[1, 1]
            b23 = b2s[1, 2]
            b24 = b2s[1, 3]
            b33 = b2s[2, 2]
            b34 = b2s[2, 3]
            b44 = b2s[3, 3]

            # can be derived from quaternion multiplication
            s11 = \
                a11*b11 - 2*a12*b12 - 2*a13*b13 - 2*a14*b14 + a22*b22 + \
                2*a23*b23 + 2*a24*b24 + a33*b33 + 2*a34*b34 + a44*b44
            s12 = \
                a11*b12 + a12*b11 + a13*b14 - a14*b13 - a12*b22 - a22*b12 - \
                a13*b23 - a23*b13 - a14*b24 - a24*b14 - a23*b24 + a24*b23 - \
                a33*b34 + a34*b33 - a34*b44 + a44*b34
            s21 = s12
            s13 = \
                a11*b13 + a13*b11 - a12*b14 + a14*b12 - a12*b23 - a23*b12 - \
                a13*b33 + a22*b24 - a24*b22 - a33*b13 - a14*b34 - a34*b14 + \
                a23*b34 - a34*b23 + a24*b44 - a44*b24
            s31 = s13
            s14 = \
                a11*b14 + a12*b13 - a13*b12 + a14*b11 - a12*b24 - a24*b12 - \
                a22*b23 + a23*b22 - a13*b34 - a34*b13 - a23*b33 + a33*b23 - \
                a14*b44 - a24*b34 + a34*b24 - a44*b14
            s41 = s14
            s22 = \
                2*a12*b12 + a11*b22 + a22*b11 + 2*a13*b24 - 2*a14*b23 + \
                2*a23*b14 - 2*a24*b13 - 2*a34*b34 + a33*b44 + a44*b33
            s23 = \
                a12*b13 + a13*b12 + a11*b23 + a23*b11 - a12*b24 + a14*b22 - \
                a22*b14 + a24*b12 + a13*b34 - a14*b33 + a33*b14 - a34*b13 + \
                a24*b34 + a34*b24 - a23*b44 - a44*b23
            s32 = s23
            s24 = \
                a12*b14 + a14*b12 + a11*b24 + a12*b23 - a13*b22 + a22*b13 - \
                a23*b12 + a24*b11 - a14*b34 + a34*b14 + a13*b44 + a23*b34 - \
                a24*b33 - a33*b24 + a34*b23 - a44*b13
            s42 = s24
            s33 = \
                2*a13*b13 + 2*a14*b23 - 2*a23*b14 + a11*b33 + a33*b11 - \
                2*a12*b34 + 2*a34*b12 - 2*a24*b24 + a22*b44 + a44*b22
            s34 = \
                a13*b14 + a14*b13 - a13*b23 + a23*b13 + a14*b24 - a24*b14 + \
                a11*b34 + a12*b33 - a33*b12 + a34*b11 + a23*b24 + a24*b23 - \
                a12*b44 - a22*b34 - a34*b22 + a44*b12
            s43 = s34
            s44 = \
                2*a14*b14 - 2*a13*b24 + 2*a24*b13 + 2*a12*b34 - 2*a23*b23 - \
                2*a34*b12 + a11*b44 + a22*b33 + a33*b22 + a44*b11

            s = np.array([[s11, s12, s13, s14],
                          [s21, s22, s23, s24],
                          [s31, s32, s33, s34],
                          [s41, s42, s43, s44]])

            return BinghamDistribution.fit_to_moment(s)

    @staticmethod
    def assert_parameters(param_m, param_z):
        """Asserts param_m and param_z to satisfy requirements of the Bingham"""
        assert isinstance(param_m, np.ndarray), \
            "m needs to be of type numpy.ndarray."
        assert isinstance(param_z, np.ndarray), \
            "z needs to be of type numpy.ndarray."

        dist_dim = param_m.shape[0]
        assert dist_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS, \
            "Not supported distribution dimension."

        # Currently we support only 2d Bingham distribution.
        assert param_m.shape == (dist_dim, dist_dim), \
            "m needs to be a square Matrix."

        assert param_z.shape == (dist_dim, ), \
            "z needs to be a vector and dimension needs to agree with m."

        # TODO: Get rid of these 2 asseritons by using properties for getting
        # and setting the location parameter m and the dispersion parameter z.
        assert param_z[-1] == 0., "Last entry of z needs to be 0."
        assert all(param_z[:-1] <= param_z[1:]), \
            "Entries of z need to be given in an ascending order."

        # Check for orthogonality of m.
        numerical_tolerance = 1e-10
        product = np.dot(param_m, param_m.T)
        diff = product - np.eye(dist_dim)
        assert np.all(np.abs(diff) < numerical_tolerance), \
            "param_m is not orthogonal."

    @staticmethod
    def decompose_parameters(param_matrix, correct_eigenvalues=True):
        """Decomposes a parameter matrix into location and dispersion part.

        The entire parameter matrix M*Z*M^T is decomposed into M and Z, where Z
        is a diagonal matrix returned as a vector.

        Parameters
        ----------
        param_matrix : array of shape(n_dim, n_dim)
            Original full parameter matrix.

        correct_eigenvalues : boolean
            Sets largest eigenvalue to 0 if true by subtracting the largest
            eigenvalue from Z (default).
        """
        (bingham_dispersion, bingham_location) = np.linalg.eig(param_matrix)
        eigval_order = np.argsort(bingham_dispersion)

        bingham_location = bingham_location[:, eigval_order]
        bingham_dispersion = bingham_dispersion[eigval_order]
        offset = 0.0

        if correct_eigenvalues:
            offset = bingham_dispersion[-1]
            bingham_dispersion = bingham_dispersion - offset

        return bingham_location, bingham_dispersion, offset

    @staticmethod
    def fit(data):
        """Fits a bingham distribution to given data.

        The implemented fitting procedure is based on the method of moments,
        i.e. we compute the empirical second moment of the data and numerically
        obtain the corresponding Bingham distribution parameters.

        Parameters
        ----------
        data : array of shape(n_points, 2)
            The samples at which the density is evaluated.

        Returns
        -------
        result : Bingham distribution object
        """

        assert isinstance(data, np.ndarray), \
            "data needs to be a np.ndarray"

        bd_dim = data.shape[1]

        assert bd_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS, \
            "Not supported Bingham distribution dimensionality."

        n_samples = data.shape[0]
        second_moment = np.dot(data.T, data)/n_samples
        return BinghamDistribution.fit_to_moment(second_moment)

    @staticmethod
    def fit_to_moment(second_moment):
        """Finds a Bingham distribution with a given second moment.

        Parameters
        ----------
        second_moment : (d x d matrix)
            matrix representing second moment.

        Returns
        -------
        b : BinghamDistribution
            the MLE estimate for a Bingham distribution given the
            scatter matrix S
        """
        assert np.allclose(second_moment, second_moment.transpose()), \
            "second moment must be symmetric"
        bd_dim = second_moment.shape[1]

        (moment_eigval, bingham_location) = np.linalg.eig(second_moment)

        # Sort eigenvalues (and corresponding eigenvectors) in asc. order.
        eigval_order = np.argsort(moment_eigval)
        bingham_location = bingham_location[:, eigval_order]
        moment_eigval = moment_eigval[eigval_order]

        logger = logging.getLogger(__name__)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("second_moment=\n%s", second_moment)
            logger.debug("moment_eigval=%s", moment_eigval)
            logger.debug("eigval_order=%s", eigval_order)
            logger.debug("bingham_location=\n%s", bingham_location)

        def mle_goal_fun(z, rhs):
            """Goal function for MLE optimizer."""

            z_param = np.append(z, 0)
            norm_const = BinghamDistribution.normalization_constant(z_param)
            norm_const_deriv \
                = BinghamDistribution.normalization_constant_deriv(z_param)

            res = (norm_const_deriv[0:(bd_dim-1)] / norm_const) \
                - rhs[0:(bd_dim-1)]
            return res

        bingham_dispersion = scipy.optimize.fsolve(
            lambda x: mle_goal_fun(x, moment_eigval), np.ones([(bd_dim-1)]))
        bingham_dispersion = np.append(bingham_dispersion, 0)
        bingham_dist = BinghamDistribution(bingham_location, bingham_dispersion)

        # Remove this bloat code.
        return bingham_dist

    @staticmethod
    def normalization_constant(param_z, mode="default", options=dict()):
        """Computes the Bingham normalization constant.

        Parameters
        ----------
        param_z : array of shape (dim)
            Diagonal entries of dispersion parameter matrix Z of the Bingham
            distribution.
        mode : string
            Method of computation (optional).
        options : dict
            Computation-method specific options.
        """
        # Gerhard Kurz, Igor Gilitschenski, Simon Julier, Uwe D. Hanebeck,
        # "Recursive Bingham Filter for Directional Estimation Involving 180
        # Degree Symmetry", Journal of Advances in Information
        # Fusion, 9(2):90 - 105, December 2014.

        bd_dim = param_z.shape[0]

        assert bd_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS \
            and param_z.ndim == 1, \
            "param_z needs to be a vector of supported dimension."

        # TODO Check structure of Z

        if bd_dim == 2:
            if mode == "default" or mode == "bessel":
                # Surface area of the unit sphere is a factor in the
                # normalization constant. The formula is taken from
                # https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
                sphere_surface_area = 2.0 * (np.pi**(bd_dim / 2.0) /
                                             scipy.special.gamma(bd_dim / 2.0))

                norm_const = (np.exp(param_z[1]) * sphere_surface_area *
                              scipy.special.iv(
                                  0, (param_z[0] - param_z[1]) / 2.0)
                              * np.exp((param_z[0] - param_z[1]) / 2.0))
                return norm_const
        elif bd_dim == 4:
            if mode == "default" or mode == "saddlepoint":
                f = BinghamDistribution.__norm_const_saddlepoint(
                    np.sort(-param_z)+1)
                f *= np.exp(1)
                return f[2]
            elif mode == "numerical":
                param_z_diag = np.diag(param_z)

                def bd_likelihood(x):
                    return np.exp(np.dot(x, np.dot(param_z_diag, x)))

                def integrand(phi1, phi2, phi3):
                    sp1 = np.sin(phi1)
                    sp2 = np.sin(phi2)
                    return bd_likelihood(np.array([
                        sp1 * sp2 * np.sin(phi3),
                        sp1 * sp2 * np.cos(phi3),
                        sp1 * np.cos(phi2),
                        np.cos(phi1)
                    ])) * (sp1 ** 2.) * sp2

                norm_const = integrate.tplquad(
                    integrand,
                    0.0, 2.0 * np.pi,  # phi3
                    lambda x: 0.0, lambda x: np.pi,  # phi2
                    lambda x, y: 0.0, lambda x, y: np.pi,  # phi1
                    **options
                )

                return norm_const[0]

        sys.exit("Invalid computation mode / dimension combination.")

    @staticmethod
    def normalization_constant_deriv(param_z, mode="default"):
        """Computes the derivatives (w.r.t. Z) of the normalization constant.

        Parameters
        ----------
        param_z : array of shape (dim)
            Diagonal entries of dispersion parameter matrix Z of the Bingham
            distribution.
        mode : string
            Method of computation (optional).
        """

        bd_dim = param_z.shape[0]
        assert bd_dim in BinghamDistribution.IMPLEMENTED_DIMENSIONS \
            and param_z.ndim == 1, \
            "param_z needs to be a vector of supported dimension."

        derivatives = np.zeros(bd_dim)
        if bd_dim == 2 and mode == "default":
            derivatives = np.zeros(2)
            z_param_diff = (param_z[0] - param_z[1]) / 2.0
            z_param_mean = (param_z[0] + param_z[1]) / 2.0
            b1 = scipy.special.iv(1, z_param_diff)
            b0 = scipy.special.iv(0, z_param_diff)
            derivatives[0] = np.pi * np.exp(z_param_mean) * (b1 + b0)
            derivatives[1] = np.pi * np.exp(z_param_mean) * (-b1 + b0)
        elif bd_dim == 4 and mode == "quad":
            def bd_deriv_likelihood(x, j):
                return x[j]**2 * np.exp(np.dot(x, np.dot(np.diag(param_z), x)))

            for i in range(0, bd_dim):
                derivatives[i] = integrate.tplquad(
                    lambda phi1, phi2, phi3:
                    bd_deriv_likelihood(np.flip(np.array([
                        np.cos(phi1),
                        np.sin(phi1) * np.cos(phi2),
                        np.sin(phi1) * np.sin(phi2) * np.cos(phi3),
                        np.sin(phi1) * np.sin(phi2) * np.sin(phi3),
                    ])), i) * (np.sin(phi1) ** 2.) * np.sin(phi2),
                    0.0, 2.0 * np.pi,  # phi3
                    lambda x: 0.0, lambda x: np.pi,  # phi2
                    lambda x, y: 0.0, lambda x, y: np.pi  # phi1
                )[0]
        else:
            if mode == "default" or mode == "saddlepoint":
                derivatives = np.zeros(bd_dim)
                for i in range(0, bd_dim):
                    modz = np.concatenate((param_z[0:i + 1],
                                           np.array([param_z[i]]),
                                           param_z[i:bd_dim + 1]))
                    t = BinghamDistribution.__norm_const_saddlepoint(
                        np.sort(-modz) + 1)
                    t *= np.exp(1) / (2 * np.pi)
                    derivatives[i] = t[2]
            else:
                sys.exit("No such computation mode.")

        return derivatives

    ##########################
    # Private Static Methods #
    ##########################
    @staticmethod
    def __xi2cgfderiv(t, dim, la, derriv):
        """Calculates first 4 derivatives of the cumulant generating function"""
        res = [0] * 4
        for i in range(dim):
            if i == derriv:
                scale = 3.0
            else:
                scale = 1.0
            res[0] += scale*0.5/(la[i]-t)
            res[1] += scale*0.5/((la[i]-t)*(la[i]-t))
            res[2] += scale*1/((la[i]-t)*(la[i]-t)*(la[i]-t))
            res[3] += scale*3/((la[i]-t)*(la[i]-t)*(la[i]-t)*(la[i]-t))
        return res

    @staticmethod
    def __find_root_newton(dim, la, min_el):
        """Root finding algorithm using Newton's Method"""
        prec = 1E-10  # Precision
        x = min_el - 0.5  # Upper bound for initial evaluation point
        i = 0
        while True:
            val = BinghamDistribution.__xi2cgfderiv(x, dim, la, -1)
            val[0] -= 1
            x += -val[0] / val[1]
            i += 1
            if not ((val[0] > prec or val[0] < -prec) and i < 1000):
                break
        return x

    @staticmethod
    def __find_multiple_roots_newton(dim, la, min_el):
        """Multiple roots finding algorithm using Newton's Method"""
        prec = 1E-10
        ubound = min_el - 0.5
        retval = [ubound] * (dim + 1)  # set starting value of Newton method
        i = 0
        while True:
            err = 0
            # Iterate over the Norm const and each partial derivative
            for j in range(dim + 1):
                v0 = 0
                v1 = 0
                for k in range(dim):
                    if k != j - 1:
                        v0 += 0.5 / (la[k] - retval[j])
                        v1 += 0.5 / ((la[k] - retval[j]) * (la[k]-retval[j]))
                    else:
                        v0 += 3 * 0.5/(la[k] - retval[j])
                        v1 += 3 * 0.5/((la[k] - retval[j]) * (la[k]-retval[j]))
                v0 -= 1  # because we want to solve K(t)=1
                err += abs(v0)
                retval[j] += -v0 / v1  # Newton iteration
            i += 1
            if not (err > prec and i < 1000):
                break
        return retval

    @staticmethod
    def __norm_const_saddlepoint(eigval, deriv=False):
        """ Saddlepoint based approximation of the normalization constant. """

        assert isinstance(eigval, np.ndarray), \
            "input needs to be of type numpy.ndarray."
        assert eigval.ndim == 1, \
            "input needs to be a vector"

        dim = eigval.shape[0]
        min_el = np.amin(eigval)
        result = np.zeros(3)
        derivatives = {}
        la = eigval
        scale_factor = 1.0
        if min_el <= 0:
            la = eigval - (min_el - 0.1)
            scale_factor = np.exp(-min_el + 0.1)
            min_el = 0.1
        if deriv:
            r = BinghamDistribution.__find_multiple_roots_newton(
                dim, la, min_el)
            hk = BinghamDistribution.__xi2cgfderiv(r[0], dim, la, -1)
            t = (1.0 / 8 * (hk[3] / (hk[1] * hk[1])) - 5.0 / 24 *
                 (hk[2] * hk[2] / (hk[1] * hk[1] * hk[1])))
            result[0] = (np.sqrt(2 * pow(np.pi, dim - 1)) * np.exp(-r[0]) /
                         np.sqrt(hk[1]) * scale_factor)

            for i in range(dim):
                result[0] /= np.sqrt(la[i] - r[0])

            result[1] = result[0] * (1 + t)
            result[2] = result[0] * np.exp(t)

            for i in range(dim):
                hk = BinghamDistribution.__xi2cgfderiv(r[i + 1], dim, la, i)

                t = (1.0 / 8 * (hk[3] / (hk[1] * hk[1])) - 5.0 / 24 *
                     (hk[2] * hk[2] / (hk[1] * hk[1] * hk[1])))
                derivatives[3*i] = (np.sqrt(2*pow(np.pi, dim+1))*np.exp(-r[i+1])
                                    / (np.sqrt(hk[1]) * 2 * np.pi) *
                                    scale_factor)
                for j in range(dim):
                    if j != i:
                        derivatives[3 * i] /= np.sqrt(la[j] - r[i + 1])
                    else:
                        derivatives[3 * i] /= pow(np.sqrt(la[j] - r[i + 1]), 3)

                derivatives[3 * i + 1] = derivatives[3 * i] * (1 + t)
                derivatives[3 * i + 2] = derivatives[3 * i] * np.exp(t)
                return result, derivatives
        else:
            r = BinghamDistribution.__find_root_newton(dim, la, min_el)
            hk = BinghamDistribution.__xi2cgfderiv(r, dim, la, -1)
            t = (1.0 / 8 * (hk[3] / (hk[1] * hk[1])) - 5.0 / 24 *
                 (hk[2] * hk[2] / (hk[1] * hk[1] * hk[1])))
            result[0] = (np.sqrt(2 * pow(np.pi, dim - 1)) * np.exp(-r) /
                         np.sqrt(hk[1]) * scale_factor)

            for i in range(dim):
                result[0] /= np.sqrt(la[i] - r)
            result[1] = result[0] * (1 + t)
            result[2] = result[0] * np.exp(t)
            return result
