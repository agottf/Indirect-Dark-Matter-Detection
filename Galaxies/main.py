# This is a copy of the code by Alex Drlica-Wagner
# This code shows how to load
# the bin-by-bin likelihood functions and gamma-ray
# spectrum and compute spectrally-dependent limits
# the DM annihilation cross section.
# NOTE: This script does not incorporate uncertainties
# on the measured J-factor

# Copyright (C) 2013 Alex Drlica-Wagner <kadrlica@fnal.gov>
# See <http://www-glast.stanford.edu/pub_data/1048/example3.py>
# for the code I am copying

import numpy as np
# import pylab as plt """It's bad practice to use pylab now"""
# import matplotlib

from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import brentq


def eflux(spectrum, emin=1e2, emax=1e5, quiet=False):
    """Integrate a generic spectrum, multiplied by E, to get the energy flux."""
    espectrum = lambda e: spectrum(e) * e
    tol = min(espectrum(emin), espectrum(emax)) * 1e-10
    try:
        return quad(espectrum, emin, emax, epsabs=tol, full_output=True)[0]
    except Exception as msg:
        print('Numerical error "%s" when calculating integral flux.' % msg)
        return np.nan


# Keep numpy from complaining about dN/dE = 0...
np.seterr(divide='ignore', invalid='ignore')

# Analyze Draco with 100 GeV b-bbar spectrum.
likefile = 'http://www-glast.stanford.edu/pub_data/1048/like_draco.txt'
specfile = 'specfile.txt'
# J factor of Draco (*NO UNCERTAINTY INCLUDED*)
jfactor = 10 ** 18.83

# Spectral file created assuming J=10^18 GeV^2/cm^5, sigmav=1e-25 cm^3/s
j0 = 1e18
sigmav0 = 1e-25

# Load the likelihood.
data = np.loadtxt(likefile, unpack=True)
emins, emaxs = np.unique(data[0]), np.unique(data[1])
ebin = np.sqrt(emins * emaxs)
efluxes = data[2].reshape(len(emins), -1)
logLikes = data[3].reshape(len(emins), -1)

# Load the spectrum.
energy, dnde = np.loadtxt(specfile, unpack=True)
log_energy = np.log10(energy)
log_dnde = np.log10(dnde)
log_interp = interp1d(log_energy, log_dnde)
spectrum = lambda e: np.nan_to_num(10 ** (log_interp(np.log10(e))))

# Predict energy flux from nominal spectral values
pred = np.array([eflux(spectrum, e1, e2) for e1, e2 in zip(emins, emaxs)])

# Interpolated log-likelihood in each energy bin
likes = [interp1d(f, l - l.max()) for f, l in zip(efluxes, logLikes)]

# Global log-likelihood summed over energy bins
like = lambda c: sum([lnlfn(c * p) for lnlfn, p in zip(likes, pred)])

# Scan range for global log-likelihood
epsilon = 1e-4  # Just to make sure we stay within the interpolation range
xmin = epsilon
xmax = np.log10(efluxes.max() / efluxes[efluxes > 0].min()) - epsilon
x = np.logspace(xmin, xmax, 250)

norm0 = efluxes[efluxes > 0].min() / pred.max()
norms = norm0 * x

# Global log-likelihood using nominal value of nuisance parameter
lnl = np.array([like(n) for n in norms])

# Convert global log-likelihood back into physical units
sigmav = j0 / jfactor * sigmav0 * norms
lnlfn = interp1d(sigmav, lnl)
mle = lnl.max()
sigmav_mle = sigmav[lnl.argmax()]
delta = 2.71 / 2
limit = brentq(lambda x: lnlfn(x) - mle + delta, sigmav_mle, sigmav.max(), xtol=1e-10 * sigmav_mle)
print("Upper Limit: %.5e" % limit)
