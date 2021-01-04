
from functools import lru_cache
from dataclasses import dataclass

import numpy as np

from scipy.special import ndtr, ndtri
from scipy.optimize import brentq
from .distributions import hypergeom
from .stats import fisher_exact


def _sample_odds_ratio(table):
    """
    Given a table [[a, b], [c, d]], compute a*d/(b*c).

    Return nan if the numerator and denominator are 0.
    Return inf if just the denominator is 0.
    """
    # table must be a 2x2 numpy array.
    if table[1, 0] > 0 and table[0, 1] > 0:
        oddsratio = table[0, 0] * table[1, 1] / (table[1, 0] * table[0, 1])
    elif table[0, 0] == 0 or table[1, 1] == 0:
        oddsratio = np.nan
    else:
        oddsratio = np.inf
    return oddsratio


def _hypergeom_support_bounds(total, ngood, nsample):
    """
    Returns the inclusive bounds of the support.
    """
    nbad = total - ngood
    return max(0, nsample - nbad), min(nsample, ngood)


@lru_cache()
def _hypergeom_support_and_logpmf(total, ngood, nsample):
    """
    The support and the log of the PMF of the hypergeometric distribution.
    """
    lo, hi = _hypergeom_support_bounds(total, ngood, nsample)
    support = np.arange(lo, hi + 1)
    logpmf = hypergeom._logpmf(support, total, ngood, nsample)
    return support, logpmf


def _nc_hypergeom_support_and_pmf(lognc, total, ngood, nsample):
    """
    The support and the PMF of the noncentral hypergeometric distribution.

    The first argument is the log of the noncentrality parameter.
    """
    # Get the support and log of the PMF for the corresponding
    # hypergeometric distribution.
    support, hg_logpmf = _hypergeom_support_and_logpmf(total, ngood, nsample)

    # Calculate the PMF using the ideas from
    #   Bruce Barret (2017), A note on exact calculation of the non central
    #   hypergeometric distribution, Communications in Statistics - Theory
    #   and Methods, 46:13, 6737-6741, DOI: 10.1080/03610926.2015.1134573
    #
    # Create the log of the PMF from the log of the PMF of the hypergeometric
    # distribution.
    nchg_logpmf = hg_logpmf + lognc * support

    # Exponentiation after subtracting the max to get the "quasi" PMF (not
    # correctly normalized yet).
    nchg_qpmf = np.exp(nchg_logpmf - nchg_logpmf.max())

    # Normalize to the actual PMF of the noncentral hypergeometric dist.
    nchg_pmf = nchg_qpmf / nchg_qpmf.sum()

    return support, nchg_pmf


def _nc_hypergeom_mean(lognc, total, ngood, nsample):
    """
    Expected value of Fisher's noncentral hypergeometric distribution.

    lognc is the logarithm of the noncentrality parameter.  The remaining
    parameters are the same as those of the hypergeometric distribution.
    """
    # Explicit check for the edge cases.  (XXX These checks might not be
    # necessary, but double-check that we never call this function with +/- inf
    # before removing.  Maybe we'll get these values here inadvertently because
    # of overflow or underflow.)
    if lognc == -np.inf:
        return _hypergeom_support_bounds(total, ngood, nsample)[0]
    if lognc == np.inf:
        return _hypergeom_support_bounds(total, ngood, nsample)[1]

    support, nchg_pmf = _nc_hypergeom_support_and_pmf(lognc, total,
                                                      ngood, nsample)

    # Compute the expected value of the distribution.
    mean = support.dot(nchg_pmf)
    return mean


def _solve(func):
    """
    Solve func(lognc) = 0.  func must be an increasing function.
    """
    # We could just as well call the variable `x` instead of `lognc`, but we
    # always call this function with functions for which lognc (the log of
    # the noncentrality parameter) is the variable for which we are solving.
    # It also helps to remind that the linear search for a bracketing interval
    # is actually working with logarithms of the variable of interest, so
    # changing lognc by 5 corresponds to changing nc by a multiplicative
    # factor of exp(5) ~= 148.  While the strategy used here would be terrible
    # as a general purpose solver, here it works well.
    lognc = 0
    value = func(lognc)
    if value == 0:
        return lognc

    # Find a bracketing interval.
    if value > 0:
        lognc -= 5
        while func(lognc) > 0:
            lognc -= 5
        lo = lognc
        hi = lognc + 5
    else:
        lognc += 5
        while func(lognc) < 0:
            lognc += 5
        lo = lognc - 5
        hi = lognc

    # lo and hi bracket the solution for lognc.
    lognc = brentq(func, lo, hi, xtol=1e-13)
    return lognc


def _nc_hypergeom_mean_inverse(x, total, ngood, nsample):
    """
    For the given total, ngood, and nsample, find the noncentrality
    parameter of Fisher's noncentral hypergeometric distribution whose
    mean is x.
    """
    lognc = _solve(lambda lognc: _nc_hypergeom_mean(lognc, total, ngood,
                                                    nsample) - x)
    return np.exp(lognc)


def _nc_hypergeom_cdf(x, lognc, total, ngood, nsample):
    """
    CDF of the noncentral hypergeometric distribution.
    """
    lo, hi = _hypergeom_support_bounds(total, ngood, nsample)
    if lognc == -np.inf:
        return 1.0*(x >= lo)
    elif lognc == np.inf:
        return 1.0*(x >= hi)

    support, nchg_pmf = _nc_hypergeom_support_and_pmf(lognc, total,
                                                      ngood, nsample)
    return nchg_pmf[support <= x].sum()


def _nc_hypergeom_sf(x, lognc, total, ngood, nsample):
    """
    Survival function of the noncentral hypergeometric distribution.
    """
    lo, hi = _hypergeom_support_bounds(total, ngood, nsample)
    if lognc == -np.inf:
        return 1.0*(x < lo)
    elif lognc == np.inf:
        return 1.0*(x < hi)

    support, nchg_pmf = _nc_hypergeom_support_and_pmf(lognc, total,
                                                      ngood, nsample)
    return nchg_pmf[support > x].sum()


def _hypergeom_params_from_table(table):
    x = table[0, 0]
    total = table.sum()
    ngood = table[0].sum()
    nsample = table[:, 0].sum()
    return x, total, ngood, nsample


def _ci_upper(table, alpha):
    """
    Compute the upper end of the confidence interval.
    """
    if _sample_odds_ratio(table) == np.inf:
        return np.inf

    x, total, ngood, nsample = _hypergeom_params_from_table(table)

    # _nc_hypergeom_cdf is a decreasing function of lognc, so we negate
    # it in the lambda expression.
    lognc = _solve(lambda lognc: -_nc_hypergeom_cdf(x, lognc, total,
                                                    ngood, nsample) + alpha)
    return np.exp(lognc)


def _ci_lower(table, alpha):
    """
    Compute the lower end of the confidence interval.
    """
    if _sample_odds_ratio(table) == 0:
        return 0

    x, total, ngood, nsample = _hypergeom_params_from_table(table)

    lognc = _solve(lambda lognc: _nc_hypergeom_sf(x - 1, lognc, total,
                                                  ngood, nsample) - alpha)
    return np.exp(lognc)


def _conditional_oddsratio(table):
    """
    Conditional MLE of the odds ratio for the 2x2 contingency table.
    """
    x, total, ngood, nsample = _hypergeom_params_from_table(table)
    lo, hi = _hypergeom_support_bounds(total, ngood, nsample)

    # Check if x is at one of the extremes of the support.  If so, we know
    # the odds ratio is either 0 or inf.
    if x == lo:
        # x is at the low end of the support.
        return 0
    if x == hi:
        # x is at the high end of the support.
        return np.inf

    nc = _nc_hypergeom_mean_inverse(x, total, ngood, nsample)
    return nc


def _conditional_oddsratio_ci(table, confidence_level=0.95,
                              alternative='two-sided'):
    """
    Conditional exact confidence interval for the odds ratio.

    This function implements Cornfield's "exact confidence limits",
    as explained in section 2 of the paper:

        J. Cornfield (1956), A statistical problem arising from
        retrospective studies. In Neyman, J. (ed.), Proceedings of
        the Third Berkeley Symposium on Mathematical Statistics and
        Probability 4, pp. 135-148.

    """
    if alternative == 'two-sided':
        alpha = 0.5*(1 - confidence_level)
        lower = _ci_lower(table, alpha)
        upper = _ci_upper(table, alpha)
    elif alternative == 'less':
        lower = 0.0
        upper = _ci_upper(table, 1 - confidence_level)
    else:
        # alternative == 'greater'
        lower = _ci_lower(table, 1 - confidence_level)
        upper = np.inf

    return lower, upper


def _sample_odds_ratio_ci(table, confidence_level=0.95,
                          alternative='two-sided'):
    oddsratio = _sample_odds_ratio(table)
    log_or = np.log(oddsratio)
    se = np.sqrt((1/table).sum())
    if alternative == 'less':
        z = ndtri(confidence_level)
        loglow = -np.inf
        loghigh = log_or + z*se
    elif alternative == 'greater':
        z = ndtri(confidence_level)
        loglow = log_or - z*se
        loghigh = np.inf
    else:
        # alternative is 'two-sided'
        z = ndtri(0.5*confidence_level + 0.5)
        loglow = log_or - z*se
        loghigh = log_or + z*se

    return np.exp(loglow), np.exp(loghigh)


@dataclass
class ConfidenceInterval:
    low: float
    high: float


@dataclass
class OddsRatioResult:
    """
    Result of `scipy.stats.contingency.odds_ratio`.

    Attributes
    ----------
    table : numpy.ndarray
        The table that was passed to `odds_ratio`.
    kind : str
        The ``kind`` that was passed to `odds_ratio`. This will be
        either ``'conditional'`` or ``'sample'``.
    alternative : str
        The ``alternative`` that was passed to `odds_ratio`.  This will
        be ``'two-sided'``, ``'less'`` or ``'greater'``.
    odds_ratio : float
        The computed odds ratio.
    pvalue : float
        The p-value of the estimate of the odds ratio.

    Methods
    -------
    odds_ratio_ci(confidence_level=0.95)
        Compute the confidence interval for the odds ratio.
    """

    table: np.ndarray
    kind: str
    alternative: str
    odds_ratio: float
    pvalue: float

    def odds_ratio_ci(self, confidence_level=0.95):
        """
        Confidence inteval for the odds ratio.

        Parameters
        ----------
        confidence_level: float
            Desired confidence level for the confidence interval.
            The value must be given as a fraction between 0 and 1.
            Default is 0.95 (meaning 95%).

        Returns
        -------
        ci : ``ConfidenceInterval`` instance
            The confidence interval, represented as an object with
            attributes ``low`` and ``high``.

        Notes
        -----
        When ``kind`` is `'conditional'`, the limits of the confidence
        interval are the conditional "exact confidence limits" as defined in
        section 2 of Cornfield [2]_, and originally described by Fisher [1]_.
        The conditional odds ratio and confidence interval are also discussed
        in Section 4.1.2 of the text by Sahai and Khurshid [3]_.

        When ``kind`` is ``'sample'``, the confidence interval is computed
        under the assumption that the logarithm of the odds ratio is normally
        distributed with standard error given by::

            se = sqrt(1/a + 1/b + 1/c + 1/d)

        where ``a``, ``b``, ``c`` and ``d`` are the elements of the
        contingency table.  (See, for example, [3]_, section 3.1.3.2,
        or [4]_, section 2.3.3).

        References
        ----------
        .. [1] R. A. Fisher (1935), The logic of inductive inference,
               Journal of the Royal Statistical Society, Vol. 98, No. 1,
               pp. 39-82.
        .. [2] J. Cornfield (1956), A statistical problem arising from
               retrospective studies. In Neyman, J. (ed.), Proceedings of
               the Third Berkeley Symposium on Mathematical Statistics
               and Probability 4, pp. 135-148.
        .. [3] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
               Methods, Techniques, and Applications, CRC Press LLC, Boca
               Raton, Florida.
        .. [4] Alan Agresti, An Introduction to Categorical Data Analyis
               (second edition), Wiley, Hoboken, NJ, USA (2007).
        """
        if self.kind == 'conditional':
            ci = self._conditional_odds_ratio_ci(confidence_level)
        else:
            ci = self._sample_odds_ratio_ci(confidence_level)
        return ci

    def _conditional_odds_ratio_ci(self, confidence_level=0.95):
        """
        Confidence interval for the conditional odds ratio.
        """
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        table = self.table
        if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
            # If both values in a row or column are zero, the p-value is 1,
            # the odds ratio is NaN and the confidence interval is (0, inf).
            ci = (0, np.inf)
        else:
            ci = _conditional_oddsratio_ci(table,
                                           confidence_level=confidence_level,
                                           alternative=self.alternative)
        return ConfidenceInterval(low=ci[0], high=ci[1])

    def _sample_odds_ratio_ci(self, confidence_level=0.95):
        """
        Confidence interval for the sample odds ratio.
        """
        if confidence_level < 0 or confidence_level > 1:
            raise ValueError('confidence_level must be between 0 and 1')

        table = self.table
        if 0 in table.sum(axis=0) or 0 in table.sum(axis=1):
            # If both values in a row or column are zero, the p-value is 1,
            # the odds ratio is NaN and the confidence interval is (0, inf).
            ci = (0, np.inf)
        else:
            ci = _sample_odds_ratio_ci(table,
                                       confidence_level=confidence_level,
                                       alternative=self.alternative)
        return ConfidenceInterval(low=ci[0], high=ci[1])


def odds_ratio(table, kind='conditional', alternative='two-sided'):
    r"""
    Compute the odds ratio for a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    kind : str, optional
        Which kind of odds ratio to compute, either the sample
        odds ratio (``kind='sample'``) or the conditional odds ratio
        (``kind='conditional'``).  Default is ``'conditional'``.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided'
        * 'less': one-sided
        * 'greater': one-sided

    Returns
    -------
    result : `OddsRatioResult` instance
        The returned object has two computed attributes:

        * ``odds_ratio``, float,

          * If ``kind`` is ``'sample'``, this is
            ``table[0, 0]*table[1, 1]/(table[0, 1]*table[1, 0])``.
            This is the prior odds ratio and not a posterior estimate.
          * If ``kind`` is ``'conditional'``, this is the conditional
            maximum likelihood estimate for the odds ratio. It is
            the noncentrality parameter of Fisher's noncentral
            hypergeometric distribution with the same hypergeometric
            parameters as ``table`` and whose mean is ``table[0, 0]``.

        * ``pvalue``, float

          The p-value associated with the computed odds ratio.

          * If ``kind`` is ``'sample'``, the p-value is based on the
            normal approximation to the distribution of the log of
            the sample odds ratio.
          * If ``kind`` is ``'conditional'``, the p-value is computed
            by `scipy.stats.fisher_exact`.

        The object also stores the input arguments ``table``, ``kind``
        and ``alternative`` as attributes.

        The object has the method ``odds_ratio_ci`` that computes
        the confidence interval of the odds ratio.

    References
    ----------
    .. [1] J. Cornfield (1956), A statistical problem arising from
           retrospective studies. In Neyman, J. (ed.), Proceedings of
           the Third Berkeley Symposium on Mathematical Statistics and
           Probability 4, pp. 135-148.
    .. [2] H. Sahai and A. Khurshid (1996), Statistics in Epidemiology:
           Methods, Techniques, and Applications, CRC Press LLC, Boca
           Raton, Florida.

    Examples
    --------
    In epidemiology, individuals are classified as "exposed" or
    "unexposed" to some factor or treatment. If the occurrence of some
    illness is under study, those who have the illness are often
    classifed as "cases", and those without it are "noncases".  The
    counts of the occurrences of these classes gives a contingency
    table::

                    exposed    unexposed
        cases          a           b
        noncases       c           d

    The sample odds ratio may be written ``(a/c) / (b/d)``.  ``a/c`` can
    be interpreted as the odds of a case occurring in the exposed group,
    and ``b/d`` as the odds of a case occurring in the unexposed group.
    The sample odds ratio is the ratio of these odds.  If the odds ratio
    is greater than 1, it suggests that there is a positive association
    between being exposed and being a case.

    Interchanging the rows or columns of the contingency table inverts
    the odds ratio, so it is import to understand the meaning of labels
    given to the rows and columns of the table when interpreting the
    odds ratio.  This is in contrast to the p-value, which is invariant
    under interchanging the rows or columns.

    Consider a hypothetical example where it is hypothesized that
    exposure to a certain chemical is assocated with increased occurrence
    of a certain disease.  Suppose we have the following table for a
    collection of 410 people::

                  exposed   unexposed
        cases         7         15
        noncases     58        472

    The question we ask is "Is exposure to the chemical associated with
    increased risk of the disease?"

    Compute the test, and first check the p-value.

    >>> from scipy.stats.contingency import odds_ratio
    >>> test = odds_ratio([[7, 15], [58, 472]])
    >>> test.pvalue
    0.009208708293019454

    The p-value suggests that the apparent association of exposure
    and cases is significant.

    >>> test.odds_ratio
    3.783668770554967

    The conditional odds ratio is larger than 1, which also suggests
    there is an association.  We can compute 95% confidence interval
    for the odds ratio:

    >>> test.odds_ratio_ci(confidence_level=0.95)
    ConfidenceInterval(low=1.2514829132265854, high=10.363493716701255)

    The 95% confidence interval for the conditional odds ratio is
    approximately (1.25, 10.4), which is strong evidence that the
    odds ratio is greater than 1.
    """
    if kind not in ['conditional', 'sample']:
        raise ValueError("kind must be 'conditional' or 'sample'.")
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("alternative must be 'two-sided', 'less' or "
                         "'greater'.")

    c = np.asarray(table)

    if c.shape != (2, 2):
        raise ValueError(f"Invalid shape {c.shape}. The input `table` must be "
                         "of shape (2, 2).")

    if not np.issubdtype(c.dtype, np.integer):
        raise ValueError("`table` must be an array of integers, but got "
                         f"type {c.dtype}")
    c = c.astype(np.int64)

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # If both values in a row or column are zero, the p-value is 1 and
        # the odds ratio is NaN.
        result = OddsRatioResult(table=c, kind=kind, alternative=alternative,
                                 odds_ratio=np.nan, pvalue=np.nan)
        return result

    if kind == 'sample':
        oddsratio = _sample_odds_ratio(c)
        log_or = np.log(oddsratio)
        se = np.sqrt((1/table).sum())
        if alternative == 'two-sided':
            pvalue = 2*ndtr(-abs(log_or)/se)
        elif alternative == 'less':
            pvalue = ndtr(log_or/se)
        else:
            pvalue = ndtr(-log_or/se)
    else:
        # kind is 'conditional'
        oddsratio = _conditional_oddsratio(c)
        # We can use fisher_exact to compute the p-value.
        pvalue = fisher_exact(c, alternative=alternative)[1]

    result = OddsRatioResult(table=c, kind=kind, alternative=alternative,
                             odds_ratio=oddsratio, pvalue=pvalue)
    return result
