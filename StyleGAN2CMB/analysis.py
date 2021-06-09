"""
"""
__author__ = "Manuel Blanco Valentin"

""" Basic modules """
import numpy as np
import warnings

""" This is an object used to extract the angular power spectra of maps """
class AngularPowerSpectroscope(object):
    def __init__(self, ndeg, npix, bin_factor=1.0, factor=lambda l: 1.):
        dx = np.deg2rad(ndeg / npix)
        lmax = np.pi / dx
        lstep = lmax * 2 / npix
        lmaxplot = lmax / np.sqrt(2)
        # tfac = dx / npix
        binsize = bin_factor * np.sqrt(2) * np.pi / (dx * npix)
        lbins = np.arange(binsize, lmaxplot, binsize)
        cbins = 0.5 * (lbins[0:-1] + lbins[1:])
        lx, ly = np.meshgrid(np.linspace(-lmax + lstep, lmax, npix),
                             np.linspace(-lmax + lstep, lmax, npix))
        ell = np.sqrt(lx ** 2 + ly ** 2)
        norm, bins = np.histogram(ell, bins=lbins)
        norm = norm.astype(float)
        norm[np.where(norm != 0.0)] = 1. / norm[np.where(norm != 0.0)]
        # norm = norm[None,:]

        idx, idy = np.meshgrid(np.arange(npix // 2 + 1), np.arange(npix))
        _idx = self._unwrap_fourier_2d(idx, npix).astype(int)
        _idy = self._unwrap_fourier_2d(idy, npix).astype(int)

        mask = np.array([(ell >= lbins[i]) & (ell < lbins[i + 1]) for i in range(len(lbins) - 1)])
        mask = np.transpose(mask[None,:],(0,2,3,1))

        self.mask = mask
        self.dx = dx
        self.lmax = lmax
        self.lstep = lstep
        self.lmaxplot = lmaxplot
        self.binsize = binsize
        self.lbins = lbins
        self.lx = lx
        self.ly = ly
        self.ell = ell
        self.norm = norm
        self.bin_factor = bin_factor
        self._idx = _idx
        self._idy = _idy
        self.cbins = cbins
        self.factor = factor

    def get_spectra(self, X, Y=None, factor=None, label='', verbose=True, return_phase=False):
        factor = factor if factor is not None else self.factor
        out = []
        out_phase = []
        #if verbose:
        #    progbar = Progbar(len(X) - 1)
        for i in np.array_split(np.arange(0, len(X)), len(X) // np.minimum(1000, len(X))):
            F = np.fft.rfft2(X[i], axes=(1, 2))
            FY = F
            if Y is not None:
                FY = np.fft.rfft2(Y[i], axes=(1, 2))
            _F = F * np.conj(FY)
            _F = np.real(_F[:, self._idy, self._idx, :])
            _F = _F.astype(np.float32)
            cl2 = (_F[:, :, :, 0:1] * self.mask).sum((1, 2))
            cl2 *= self.norm[None, :]
            cl2 *= factor(self.cbins)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                cl2 = np.log10(cl2)
            out.append(cl2)

            if return_phase:
                phi = np.angle(F)
                # phi = np.real(F)+1j*np.imag(FY)
                # phi = np.real(phi[:,:,self._idy,self._idx])
                # cl_phi = (phi[:,0:1]*self.mask).sum((-2,-1))
                # cl2 *= self.norm[None,:]
                # cl2 *= factor(self.cbins)
                # cl2 = np.log10(cl2)

                out_phase.append(phi)

            #progbar.add(len(i), values=[(label, max(i) / (len(X) - 1))]) if verbose else None

        if return_phase:
            return np.vstack(out), np.concatenate(out_phase)
        else:
            return np.vstack(out)

    def _unwrap_fourier_2d(self, spectrum, npix):
        unwrapped = np.zeros((npix, npix))
        halfpix = npix // 2
        spectrum_r = np.real(spectrum)

        spectrum_inv = np.concatenate((spectrum_r[(halfpix + 1):], spectrum_r[:(halfpix + 1)]))
        unwrapped[:, (halfpix - 1):] = spectrum_inv
        unwrapped[-1, :(halfpix - 1)] = spectrum_inv[-1, 1:halfpix][::-1]
        unwrapped[:-1, :(halfpix - 1)] = spectrum_inv[:-1, 1:halfpix][::-1, ::-1]

        return unwrapped
