import numpy as np

np.seterr(all='ignore')
# np.set_printoptions(threshold=sys.maxsize)


class Power(object):
    """
    Container for power spectra for each component, with any shape

    Attributes
    ----------
    c11 : :class:`~numpy.ndarray`
        Power spectral density for component 1 (any shape)
    c22 : :class:`~numpy.ndarray`
        Power spectral density for component 2 (any shape)
    cZZ : :class:`~numpy.ndarray`
        Power spectral density for component Z (any shape)
    cPP : :class:`~numpy.ndarray`
        Power spectral density for component P (any shape)
    """

    def __init__(self, c11=None, c22=None, cZZ=None, cPP=None):
        self.c11 = c11
        self.c22 = c22
        self.cZZ = cZZ
        self.cPP = cPP

    @staticmethod
    def plot_one(f, pp, name='', fig=None, fig_grid=(1, 1), plot_spot=(0, 0),
                 xlabel=True, ylabel=True):
        """Plot one cross-power spectra"""
        if not fig:
            fig = plt.gcf()
        # Plot amplitude
        ax = plt.subplot2grid((3*fig_grid[0], 1*fig_grid[1]),
                              (3*plot_spot[0]+0, plot_spot[1]+0),
                              rowspan=2)
        ax.loglog(f, np.abs(pp))
        ax.set_ylimits(0, 1)
        if ylabel:
            ax.set_ylabel('{name} PSD')
        # Plot phase
        ax = plt.subplot2grid((3*fig_grid[0], 1*fig_grid[1]),
                              (3*plot_spot[0]+2, plot_spot[1]+0))
        ax.semilogx(f, np.degrees(np.angle(pp)))
        ax.set_ylimits(-180, 180)
        if ylabel:
            ax.set_ylabel('Phase(deg)')
        if xlabel:
            ax.set_xlabel('Frequency (Hz)')

    def plot(self, f, fig=None):
        """
        Plot all power spectra

        Grid = Z1 Z2 ZP
                  12 1P
                     2P
         """
        if not fig:
            fig = plt.gcf()
        if c11 is not None:
            self.plot_one(f, c11, '1', fig, (2, 2), (0, 0), xlabel=False)
        if c22 is not None:
            self.plot_one(f, c22, '2', fig, (2, 2), (0, 1), xlabel=False,
                          ylabel=False)
        if cZZ is not None:
            self.plot_one(f, cZZ, '3', fig, (2, 2), (1, 0))
        if cPP is not None:
            self.plot_one(f, cPP, '4', fig, (2, 2), (1, 1), ylabel=False)
        plt.show()


class Cross(object):
    """
    Container for cross-power spectra for each component pairs, with any shape

    Attributes
    ----------
    c12 : :class:`~numpy.ndarray`
        Cross-power spectral density for components 1 and 2 (any shape)
    c1Z : :class:`~numpy.ndarray`
        Cross-power spectral density for components 1 and Z (any shape)
    c1P : :class:`~numpy.ndarray`
        Cross-power spectral density for components 1 and P (any shape)
    c2Z : :class:`~numpy.ndarray`
        Cross-power spectral density for components 2 and Z (any shape)
    c2P : :class:`~numpy.ndarray`
        Cross-power spectral density for components 2 and P (any shape)
    cZP : :class:`~numpy.ndarray`
        Cross-power spectral density for components Z and P (any shape)
    """

    def __init__(self, c12=None, c1Z=None, c1P=None, c2Z=None, c2P=None,
                 cZP=None):
        self.c12 = c12
        self.c1Z = c1Z
        self.c1P = c1P
        self.c2Z = c2Z
        self.c2P = c2P
        self.cZP = cZP

    @staticmethod
    def plot_one(f, cp, fig=None, fig_grid=(1, 1), plot_spot=(0, 0),
                 xlabel=True, ylabel=True):
        """Plot one cross-power spectra"""
        if not fig:
            fig = plt.gcf()
        # Plot amplitude
        ax = plt.subplot2grid((3*fig_grid[0], 1*fig_grid[1]),
                              (3*plot_spot[0]+0, plot_spot[1]+0),
                              rowspan=2)
        ax.semilogx(f, np.abs(cp))
        ax.set_ylimits(0, 1)
        if ylabel:
            ax.set_ylabel(f'{name} cross-spectra')
        # Plot phase
        ax = plt.subplot2grid((3*fig_grid[0], 1*fig_grid[1]),
                              (3*plot_spot[0]+2, plot_spot[1]+0))
        ax.semilogx(f, np.degrees(np.angle(cp)))
        ax.set_ylimits(-180, 180)
        if ylabel:
            ax.set_ylabel('Phase(deg)')
        if xlabel:
            ax.set_xlabel('Frequency (Hz)')

    def plot(self, f, fig=None):
        """
        Plot all cross-power spectra

        Grid = Z1 Z2 ZP
                  12 1P
                     2P
         """
        if not fig:
            fig = plt.gcf()
        if cZ1 is not None:
            self.plot_one(f, cZ1, 'Z-1', fig, (3, 3), (0, 0))
        if cZ2 is not None:
            self.plot_one(f, cZ2, 'Z-2', fig, (3, 3), (0, 1),
                          xlabel=False, ylabel=False)
        if cZP is not None:
            self.plot_one(f, cZP, 'Z-P', fig, (3, 3), (0, 2),
                          xlabel=False, ylabel=False)
        if c12 is not None:
            self.plot_one(f, c12, '1-2', fig, (3, 3), (1, 1))
        if c1P is not None:
            self.plot_one(f, c1P, '1-P', fig, (3, 3), (1, 2),
                          xlabel=False, ylabel=False)
        if c2P is not None:
            self.plot_one(f, c2P, '2-P', fig, (3, 3), (2, 2))
        plt.show()


class Rotation(object):
    """
    Container for rotated spectra, with any shape

    Attributes
    ----------
    cHH : :class:`~numpy.ndarray`
        Power spectral density for rotated horizontal component H (any shape)
    cHZ : :class:`~numpy.ndarray`
        Cross-power spectral density for components H and Z (any shape)
    cHP : :class:`~numpy.ndarray`
        Cross-power spectral density for components H and P (any shape)
    coh : :class:`~numpy.ndarray`
        Coherence between horizontal components
    ph : :class:`~numpy.ndarray`
        Phase of cross-power spectrum between horizontal components
    tilt : float
        Angle (azimuth) of tilt axis
    coh_value : float
        Maximum coherence
    phase_value : float
        Phase at maximum coherence
    direc : :class:`~numpy.ndarray`
        Directions for which the coherence is calculated

    """

    def __init__(self, cHH=None, cHZ=None, cHP=None, coh=None, ph=None,
                 tilt=None, coh_value=None, phase_value=None, direc=None):

        self.cHH = cHH
        self.cHZ = cHZ
        self.cHP = cHP
        self.coh = coh
        self.ph = ph
        self.tilt = tilt
        self.coh_value = coh_value
        self.phase_value = phase_value
        self.direc = direc
