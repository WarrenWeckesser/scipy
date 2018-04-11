from __future__ import division, print_function, absolute_import

import pytest
from numpy.testing import assert_equal, assert_almost_equal

from scipy.datasets import ascent, face, electrocardiogram


def test_ascent():
    assert_equal(ascent().shape, (512, 512))


def test_face():
    assert_equal(face().shape, (768, 1024, 3))


def test_electrocardiogram():
    # Test shape, dtype and stats of signal
    ecg = electrocardiogram()
    assert_equal(ecg.dtype, float)
    assert_equal(ecg.shape, (108000,))
    assert_almost_equal(ecg.mean(), -0.16510875)
    assert_almost_equal(ecg.std(), 0.5992473991177294)
