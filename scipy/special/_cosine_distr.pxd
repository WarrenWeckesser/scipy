"""
The functions

    cosine_cdf
    cosine_invcdf

defined here are the kernels for the ufuncs

    _cosine_cdf
    _cosine_invcdf

defined in scipy.special._funcs.

The ufuncs are used by the class scipy.stats.cosine_gen.

"""

from libc.math cimport sin, cos, fabs, M_PI, NAN


# p and q (below) are the coefficients in the numerator and denominator
# polynomials (resp.) of the Pade approximation of
#     f(x) = (pi + x + sin(x))/(2*pi)
# at x=-pi.  The coefficients are ordered from lowest degree to highest.
# These values are used in the function cosine_cdf_pade_approx_at_neg_pi(x).
#
# These coefficients can be derived by using mpmath as follows:
#
#     import mpmath
#
#     def f(x):
#         return (mpmath.pi + x + mpmath.sin(x)) / (2*mpmath.pi)
#
#     # Note: 40 digits might be overkill; a few more digits than the default
#     # might be sufficient.
#     mpmath.mp.dps = 40
#     ts = mpmath.taylor(f, -mpmath.pi, 20)
#     p, q = mpmath.pade(ts, 9, 10)
#
# The following are the values after converting to 64 bit floating point:
# p = [0.0,
#      0.0,
#      0.0,
#      0.026525823848649224,
#      0.0,
#      -0.0007883197097740538,
#      0.0,
#      1.0235408442872927e-05,
#      0.0,
#      -3.8360369451359084e-08]
# q = [1.0,
#      0.0,
#      0.020281047093125535,
#      0.0,
#      0.00020944197182753272,
#      0.0,
#      1.4162345851873058e-06,
#      0.0,
#      6.498171564823105e-09,
#      0.0,
#      1.6955280904096042e-11]
#

cdef inline double cosine_cdf_pade_approx_at_neg_pi(double x) nogil:
    """
    Compute the CDF of the standard cosine distribution for x close to but
    not less than -π.  A Pade approximant is used to avoid the loss of
    precision that occurs in the formula 1/2 + (x + sin(x))/(2*pi) when
    x is near -π.
    """
    cdef double h, h2, h3
    cdef double numer, denom

    # M_PI is not exactly π.  In fact, float64(π - M_PI) is
    # 1.2246467991473532e-16.  h is supposed to be x + π, so to compute
    # h accurately, we write the calculation as:
    h = (x + M_PI) + 1.2246467991473532e-16
    h2 = h*h
    h3 = h2*h
    numer = (h3*(0.026525823848649224
                 + h2*(-0.0007883197097740538
                       + h2*(1.0235408442872927e-05 
                             + h2*(-3.8360369451359084e-08)))))
    denom = (1.0 + h2*(0.020281047093125535
                       + h2*(0.00020944197182753272
                             + h2*(1.4162345851873058e-06
                                   + h2*(6.498171564823105e-09
                                         + h2*(1.6955280904096042e-11))))))
    return numer / denom


cdef inline double cosine_cdf(double x) nogil:
    """
    cosine distribution cumulative distribution function (CDF).
    """
    if x >= M_PI:
        return 1
    elif x < -M_PI:
        return 0
    elif x < -1.6:
        return cosine_cdf_pade_approx_at_neg_pi(x)
    else:
        return 0.5 + (x + sin(x))/(2*M_PI)


# The CDF of the cosine distribution is
#     p = (pi + x + sin(x)) / (2*pi),
# We want the inverse of this.
#
# Move the factor 2*pi and the constant pi to express this as
#     pi*(2*p - 1) = x + sin(x)
# Then if f(x) = x + sin(x), and g(x) is the inverse of f, we have
#     x = g(pi*(2*p - 1)).

# The coefficients in the functions _p2 and _q2 are the coefficients in the
# Pade approximation at p=0.5 to the inverse of x + sin(x).
# The following steps were used to derive these:
# 1. Find the coefficients of the Taylor polynomial of x + sin(x) at x = 0.
#    A Taylor polynomial of order 22 was used.
# 2. "Revert" the Taylor coefficients to find the Taylor polynomial of the
#    inverse.  The method for series reversion is described at
#        https://en.wikipedia.org/wiki/Bell_polynomials#Reversion_of_series
# 3. Convert the Taylor coefficients of the inverse to the (11, 10) Pade
#    approximant. The coefficients of the Pade approximant (converted to 64
#    bit floating point) in increasing order of degree are:
#      p = [0.0, 0.5,
#           0.0, -0.11602142940208726,
#           0.0, 0.009350454384541677,
#           0.0, -0.00030539712907115167,
#           0.0, 3.4900934227012284e-06,
#           0.0, -6.8448463845552725e-09]
#      q = [1.0,
#           0.0, -0.25287619213750784,
#           0.0, 0.022927496105281435,
#           0.0, -0.0008916919927321117,
#           0.0, 1.3728570152788793e-05,
#           0.0, -5.579679571562129e-08]
#
# The nonzero values in p and q are used in the functions _p2 and _q2 below.
# The functions assume that the square of the variable is passed in, and
# _p2 does not include the outermost multiplicative factor.
# So to evaluate x = invcdf(p) for a given p, the following is used:
#        y = pi*(2*p - 1)
#        x = y * _p2(y**2) / _q2(y**2)

cdef inline double _p2(double t) nogil:
    return (0.5
            + t*(-0.11602142940208726
                 + t*(0.009350454384541677
                      + t*(-0.00030539712907115167
                           + t*(3.4900934227012284e-06
                                + t*-6.8448463845552725e-09)))))

cdef inline double _q2(double t) nogil:
    return (1.0
            + t*(-0.25287619213750784
                 + t*(0.022927496105281435
                      + t*(-0.0008916919927321117
                           + t*(1.3728570152788793e-05
                                + t*-5.579679571562129e-08)))))


cdef inline double _poly_approx(double s) nogil:
    """
    Part of the asymptotic expansion of the inverse function at p=0.

    See, for example, the wikipedia article "Kepler's equation"
    (https://en.wikipedia.org/wiki/Kepler%27s_equation).  In particular, see the
    series expansion for the inverse Kepler equation when the eccentricity e is 1.
    """
    cdef double s2
    cdef double p
    #
    # p(s) = s + (1/60) * s**3 + (1/1400) * s**5 + (1/25200) * s**7 +
    #        (43/17248000) * s**9 + (1213/7207200000) * s**11 +
    #        (151439/12713500800000) * s**13 + ...
    #
    # Here we include terms up to s**13.
    s2 = s**2
    p = s*(1.0
           + s2*(1.0/60
                 + s2*(1.0/1400
                       + s2*(1.0/25200
                             + s2*(43.0/17248000
                                   + s2*(1.683039183039183e-07
                                         + s2*1.1911667949082915e-08))))))
    return p


cdef inline double cosine_invcdf(double p) nogil:
    """
    cosine distribution inverse CDF (aka percent point function).
    """
    cdef double x
    cdef double y, y2
    cdef double f0, f1, f2
    cdef int i
    cdef int sgn = 1
    cdef double prevx

    if p < 0 or p > 1:
        return NAN
    elif p <= 1e-48:
        return -M_PI
    elif p == 1:
        return M_PI

    if p > 0.5:
        p = 1.0 - p
        sgn = -1

    if p < 0.0925:
        x = _poly_approx((12*M_PI*p)**(1/3)) - M_PI
    else:
        y = M_PI*(2*p - 1)
        y2 = y**2
        x = y * _p2(y2) / _q2(y2)

    # For p < 0.0018, the asymptotic expansion at p=0 is sufficently
    # accurate that no more work is needed.  Similarly, for p > 0.42,
    # the Pade approximant is sufficiently accurate.  In between these
    # bounds, we refine the estimate with Halley's method.

    if 0.0018 < p < 0.42:
        # Several iterations of Halley's method:
        #    f(x) = pi + x + sin(x) - y,  f'(x) = 1 + cos(x), f''(x) = -sin(x)
        # where y = 2*pi*p.
        prevx = x
        for i in range(12):
            f0 = M_PI + x + sin(x) - 2*M_PI*p
            f1 = 1 + cos(x)
            f2 = -sin(x)
            x = x - 2*f0*f1/(2*f1*f1 - f0*f2)
            if fabs(x - prevx)/x < 5e-15:
                break
            prevx = x
        else:
            # This should not happen!  Extensive testing shows that the loop
            # should break before completing 12 iterations.
            x = NAN

    return sgn*x
