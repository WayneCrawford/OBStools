# import sys
from scipy.signal import spectrogram, detrend
from scipy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from obspy.core import Stream, Trace, read
from obstools.atacr import utils, plotting
from pkg_resources import resource_filename
from pathlib import Path

from .containers import Cross, Power, Rotation

np.seterr(all='ignore')
# np.set_printoptions(threshold=sys.maxsize)


class DayNoise(object):
    r"""
    A DayNoise object contains attributes that associate
    three-component raw (or deconvolved) traces, metadata information
    and window parameters. The available methods carry out the quality
    control steps and the average daily spectra for windows flagged as
    "good".

    Note
    ----
    The object is initialized with :class:`~obspy.core.Trace` objects for
    H1, H2, HZ and P components. Traces can be empty if data are not
    available. Upon saving, those traces are discarded to save disk space.

    Attributes
    ----------
    window : float
        Length of time window in seconds
    overlap : float
        Fraction of overlap between adjacent windows
    key : str
        Station key for current object
    dt : float
        Sampling distance in seconds. Obtained from ``trZ`` object
    npts : int
        Number of points in time series. Obtained from ``trZ`` object
    fs : float
        Sampling frequency (in Hz). Obtained from ``trZ`` object
    year : str
        Year for current object (obtained from UTCDateTime). Obtained from
        ``trZ`` object
    julday : str
        Julian day for current object (obtained from UTCDateTime). Obtained
        from ``trZ`` object
    ncomp : int
        Number of available components (either 2, 3 or 4). Obtained from
        non-empty ``Trace`` objects
    tf_list : Dict
        Dictionary of possible transfer functions given the available
        components.
    goodwins : list
        List of booleans representing whether a window is good (True) or
        not (False).
        This attribute is returned from the method
        :func:`~obstools.atacr.classes.DayNoise.QC_daily_spectra`
    power : :class:`~obstools.atacr.classes.Power`
        Container for daily spectral power for all available components
    cross : :class:`~obstools.atacr.classes.Cross`
        Container for daily cross spectral power for all available components
    rotation : :class:`~obstools.atacr.classes.Rotation`
        Container for daily rotated (cross) spectral power for all available
        components
    f : :class:`~numpy.ndarray`
        Frequency axis for corresponding time sampling parameters. Determined
        from method
        :func:`~obstools.atacr.classes.DayNoise.average_daily_spectra`

    Examples
    --------

    Get demo noise data as DayNoise object

    >>> from obstools.atacr import DayNoise
    >>> daynoise = DayNoise('demo')
    Uploading demo data - March 04, 2012, station 7D.M08A

    Now check its main attributes

    >>> print(*[daynoise.tr1, daynoise.tr2, daynoise.trZ, daynoise.trP], sep="\n")
    7D.M08A..1 | 2012-03-04T00:00:00.005500Z - 2012-03-04T23:59:59.805500Z | 5.0 Hz, 432000 samples
    7D.M08A..2 | 2012-03-04T00:00:00.005500Z - 2012-03-04T23:59:59.805500Z | 5.0 Hz, 432000 samples
    7D.M08A..P | 2012-03-04T00:00:00.005500Z - 2012-03-04T23:59:59.805500Z | 5.0 Hz, 432000 samples
    7D.M08A..Z | 2012-03-04T00:00:00.005500Z - 2012-03-04T23:59:59.805500Z | 5.0 Hz, 432000 samples
    >>> daynoise.window
    7200.0
    >>> daynoise.overlap
    0.3
    >>> daynoise.key
    '7D.M08A'
    >>> daynoise.ncomp
    4
    >>> daynoise.tf_list
    {'ZP': True, 'Z1': True, 'Z2-1': True, 'ZP-21': True, 'ZH': True, 'ZP-H': True}

    """

    def __init__(self, tr1=None, tr2=None, trZ=None, trP=None, window=7200.,
                 overlap=0.3, key=''):
        """
        Load data and set up lists of possible transfer functions
        """
        # Load example data if initializing empty object
        if tr1 == 'demo' or tr1 == 'Demo':
            print("Uploading demo data - March 04, 2012, station 7D.M08A")
            exmpl_path = Path(resource_filename('obstools', 'examples'))
            fn = exmpl_path / 'data' / '2012.064*.SAC'
            st = read(str(fn))
            tr1 = st.select(component='1')[0]
            tr2 = st.select(component='2')[0]
            trZ = st.select(component='Z')[0]
            trP = st.select(component='P')[0]
            window = 7200.
            overlap = 0.3
            key = '7D.M08A'

        # Check that all traces are valid Trace objects
        for tr in [tr1, tr2, trZ, trP]:
            if not isinstance(tr, Trace):
                raise(Exception("Error initializing DayNoise object - "
                                + str(tr)+" is not a Trace object"))

        # Unpack everything
        self.tr1 = tr1
        self.tr2 = tr2
        self.trZ = trZ
        self.trP = trP
        self.window = window
        self.overlap = overlap
        self.key = key

        # Get trace attributes
        self.dt = self.trZ.stats.delta
        self.npts = self.trZ.stats.npts
        self.fs = self.trZ.stats.sampling_rate
        self.year = self.trZ.stats.starttime.year
        self.julday = self.trZ.stats.starttime.julday
        self.tkey = str(self.year) + '.' + str(self.julday)

        # Get number of components for the available, non-empty traces
        ncomp = np.sum(
            [1 for tr in
             Stream(traces=[tr1, tr2, trZ, trP]) if np.any(tr.data)])
        self.ncomp = ncomp

        # Build list of available transfer functions based on the number of
        # components
        if self.ncomp == 2:
            self.tf_list = {'ZP': True, 'Z1': False, 'Z2-1': False,
                            'ZP-21': False, 'ZH': False, 'ZP-H': False,
                            'ZH-P': False}
        elif self.ncomp == 3:
            self.tf_list = {'ZP': False, 'Z1': True, 'Z2-1': True,
                            'ZP-21': False, 'ZH': True, 'ZP-H': False,
                            'ZH-P': False}
        else:
            self.tf_list = {'ZP': True, 'Z1': True, 'Z2-1': True,
                            'ZP-21': True, 'ZH': True, 'ZP-H': True,
                            'ZH-P': True}

        self.QC = False
        self.av = False

    def QC_daily_spectra(self, pd=[0.004, 0.2], tol=1.5, alpha=0.05,
                         smooth=True, fig_QC=False, debug=False, save=False,
                         form='png'):
        """
        Determine daily time windows for which the spectra are
        anomalous and should be discarded in the calculation of the
        transfer functions.

        Parameters
        ----------
        pd : list
            Frequency corners of passband for calculating the spectra
        tol : float
            Tolerance threshold. If spectrum > std*tol, window is flagged as
            bad
        alpha : float
            Confidence interval for f-test
        smooth : boolean
            Determines if the smoothed (True) or raw (False) spectra are used
        fig_QC : boolean
            Whether or not to produce a figure showing the results of the
            quality control
        debug : boolean
            Whether or not to plot intermediate steps in the QC procedure
            for debugging

        Attributes
        ----------
        goodwins : list
            List of booleans representing whether a window is good (True)
            or not (False)

        Examples
        --------

        Perform QC on DayNoise object using default values and plot final
        figure

        >>> from obstools.atacr import DayNoise
        >>> daynoise = DayNoise('demo')
        Uploading demo data - March 04, 2012, station 7D.M08A
        >>> daynoise.QC_daily_spectra(fig_QC=True)

        .. figure:: ../obstools/examples/figures/Figure_3a.png
           :align: center

        Print out new attribute of DayNoise object

        >>> daynoise.goodwins
        array([False,  True,  True,  True,  True,  True,  True,  True, False,
           False,  True,  True,  True,  True,  True,  True], dtype=bool)

        """

        # Points in window
        ws = int(self.window/self.dt)

        # Number of points to overlap
        ss = int(self.window*self.overlap/self.dt)

        # hanning window
        hanning = np.hanning(2*ss)
        wind = np.ones(ws)
        wind[0:ss] = hanning[0:ss]
        wind[-ss:ws] = hanning[ss:ws]

        # Get spectrograms for single day-long keys
        psd1 = None
        psd2 = None
        psdZ = None
        psdP = None
        f, t, psdZ = spectrogram(
            self.trZ.data, self.fs, window=wind, nperseg=ws, noverlap=ss)
        if self.ncomp == 2 or self.ncomp == 4:
            f, t, psdP = spectrogram(
                self.trP.data, self.fs, window=wind, nperseg=ws, noverlap=ss)
        if self.ncomp == 3 or self.ncomp == 4:
            f, t, psd1 = spectrogram(
                self.tr1.data, self.fs, window=wind, nperseg=ws, noverlap=ss)
            f, t, psd2 = spectrogram(
                self.tr2.data, self.fs, window=wind, nperseg=ws, noverlap=ss)

        if fig_QC:
            if self.ncomp == 2:
                self._plot_one_QC(t, f, (psdZ, psdP), ('Z', 'P'),
                                  '.specgram_Z.P.', save, form)

            elif self.ncomp == 3:
                self._plot_one_QC(t, f, (psd1, psd2, psdZ), ('H1', 'H2', 'Z'),
                                  '.specgram_H1.H2.Z.', save, form)

            else:
                self._plot_one_QC(t, f, (psd1, psd2, psdZ, psdP),
                                  ('H1', 'H2', 'Z', 'P'),
                                  '.specgram_H1.H2.Z.P.', save, form)

        # Select bandpass frequencies
        ff = (f > pd[0]) & (f < pd[1])

        if np.sum([(psd == 0.).any() for psd in
                   [psd1, psd2, psdZ, psdP] if psd is not None]) > 0.:
            smooth = True

        if smooth:
            # Smooth out the log of the PSDs
            sl_psd1 = None
            sl_psd2 = None
            sl_psdZ = None
            sl_psdP = None
            sl_psdZ = utils.smooth(np.log(psdZ, where=(psdZ > 0.)), 50, axis=0)
            if self.ncomp == 2 or self.ncomp == 4:
                sl_psdP = utils.smooth(
                    np.log(psdP, where=(psdP > 0.)), 50, axis=0)
            if self.ncomp == 3 or self.ncomp == 4:
                sl_psd1 = utils.smooth(
                    np.log(psd1, where=(psd1 > 0.)), 50, axis=0)
                sl_psd2 = utils.smooth(
                    np.log(psd2, where=(psd2 > 0.)), 50, axis=0)

        else:
            # Take the log of the PSDs
            sl_psd1 = None
            sl_psd2 = None
            sl_psdZ = None
            sl_psdP = None
            sl_psdZ = np.log(psdZ)
            if self.ncomp == 2 or self.ncomp == 4:
                sl_psdP = np.log(psdP)
            if self.ncomp == 3 or self.ncomp == 4:
                sl_psd1 = np.log(psd1)
                sl_psd2 = np.log(psd2)

        # Remove mean of the log PSDs
        dsl_psdZ = sl_psdZ[ff, :] - np.mean(sl_psdZ[ff, :], axis=0)
        if self.ncomp == 2:
            dsl_psdP = sl_psdP[ff, :] - np.mean(sl_psdP[ff, :], axis=0)
            dsls = [dsl_psdZ, dsl_psdP]
        elif self.ncomp == 3:
            dsl_psd1 = sl_psd1[ff, :] - np.mean(sl_psd1[ff, :], axis=0)
            dsl_psd2 = sl_psd2[ff, :] - np.mean(sl_psd2[ff, :], axis=0)
            dsls = [dsl_psd1, dsl_psd2, dsl_psdZ]
        else:
            dsl_psd1 = sl_psd1[ff, :] - np.mean(sl_psd1[ff, :], axis=0)
            dsl_psd2 = sl_psd2[ff, :] - np.mean(sl_psd2[ff, :], axis=0)
            dsl_psdP = sl_psdP[ff, :] - np.mean(sl_psdP[ff, :], axis=0)
            dsls = [dsl_psd1, dsl_psd2, dsl_psdZ, dsl_psdP]

        # plot log PSDs
        plt.figure(2)
        if self.ncomp == 2:
            self._plot_sl_psds(f, (sl_psdZ, sl_psdP), ('g', 'k'))
        elif self.ncomp == 3:
            self._plot_sl_psds(f, (sl_psdZ, sl_psd1, sl_psd2), ('r', 'b', 'g'))
        else:
            self._plot_sl_psds(f, (sl_psdZ, sl_psd1, sl_psd2, sl_psdP),
                               ('r', 'b', 'g', 'k'))
        if debug:
            plt.show()

        # Cycle through to kill high-std-norm windows
        moveon = False
        goodwins = np.repeat([True], len(t))
        indwin = np.argwhere(goodwins == True)

        while moveon == False:

            ubernorm = np.empty((self.ncomp, np.sum(goodwins)))
            for ind_u, dsl in enumerate(dsls):
                normvar = np.zeros(np.sum(goodwins))
                for ii, tmp in enumerate(indwin):
                    ind = np.copy(indwin)
                    ind = np.delete(ind, ii)
                    normvar[ii] = norm(np.std(dsl[:, ind], axis=1), ord=2)
                ubernorm[ind_u, :] = np.median(normvar) - normvar

            penalty = np.sum(ubernorm, axis=0)

            plt.figure(4)
            for i in range(self.ncomp):
                plt.plot(range(0, np.sum(goodwins)), detrend(
                    ubernorm, type='constant')[i], 'o-')
            if debug:
                plt.show()
            else:
                plt.close('all')
            plt.figure(5)
            plt.plot(range(0, np.sum(goodwins)),
                     np.sum(ubernorm, axis=0), 'o-')
            if debug:
                plt.show()
            else:
                plt.close('all')

            kill = penalty > tol*np.std(penalty)
            if np.sum(kill) == 0:
                self.goodwins = goodwins
                moveon = True

            trypenalty = penalty[np.argwhere(kill == False)].T[0]

            if utils.ftest(penalty, 1, trypenalty, 1) < alpha:
                goodwins[indwin[kill == True]] = False
                indwin = np.argwhere(goodwins == True)
                moveon = False
            else:
                moveon = True

        self.goodwins = goodwins

        if fig_QC:
            power = Power(sl_psd1, sl_psd2, sl_psdZ, sl_psdP)
            plot = plotting.fig_QC(
                f, power, goodwins, self.ncomp, key=self.key)

            # Save or show figure
            if save:
                fname = self.key + '.' + self.tkey + '.' + 'QC' + form
                if isinstance(save, Path):
                    fname = save / fname
                plot.savefig(
                    str(fname), dpi=300, bbox_inches='tight', format=form)
            else:
                plot.show()

        self.QC = True

    def _plot_one_QC(self, t, f, psds, names, filebase, save, form):
        plt.figure(1)
        n_psds = len(psds)
        if n_psds != len(names):
            ValueError('Different number of psds and names')
        for psd, name, i in zip(psds, names, range(n_psds)):
            plt.subplot(n_psds, 1, i+1)
            plt.pcolormesh(t, f, np.log(psd), shading='auto')
            plt.title(name, fontdict={'fontsize': 8})
        plt.xlabel('Seconds')
        plt.tight_layout()
        if save:
            fname = self.key + '.' + self.tkey + \
                filebase + form
            if isinstance(save, Path):
                fname = save / fname
            plt.savefig(
                str(fname), dpi=300, bbox_inches='tight', format=form)
        else:
            plt.show()

    @staticmethod
    def _plot_sl_psds(f, sl_psds, colors):
        n_psds = len(sl_psds)
        if not len(colors) == n_psds:
            ValueError('sl_psds and colors are not the same length')
        for sl_psd, color, i in zip(sl_psds, colors, range(n_psds)):
            plt.subplot(n_psds, 1, i+1)
            plt.semilogx(f, sl_psd, color, lw=0.5)
        plt.tight_layout()

    def average_daily_spectra(self, calc_rotation=True, fig_average=False,
                              fig_coh_ph=False, save=False, form='png'):
        """
        Method to average the daily spectra for good windows. By default, the
        method will attempt to calculate the azimuth of maximum coherence
        between horizontal components and the vertical component (for maximum
        tilt direction), and use the rotated horizontals in the transfer
        function calculations.

        Parameters
        ----------
        calc_rotation : boolean
            Whether or not to calculate the tilt direction
        fig_average : boolean
            Whether or not to produce a figure showing the average daily
            spectra
        fig_coh_ph : boolean
            Whether or not to produce a figure showing the maximum coherence
            between H and Z

        Attributes
        ----------
        f : :class:`~numpy.ndarray`
            Positive frequency axis for corresponding window parameters
        power : :class:`~obstools.atacr.classes.Power`
            Container for the Power spectra
        cross : :class:`~obstools.atacr.classes.Cross`
            Container for the Cross power spectra
        rotation : :class:`~obstools.atacr.classes.Cross`, optional
            Container for the Rotated power and cross spectra

        Examples
        --------

        Average spectra for good windows using default values and plot final
        figure

        >>> from obstools.atacr import DayNoise
        >>> daynoise = DayNoise('demo')
        Uploading demo data - March 04, 2012, station 7D.M08A
        >>> daynoise.QC_daily_spectra()
        >>> daynoise.average_daily_spectra(fig_average=True)

        .. figure:: ../obstools/examples/figures/Figure_3b.png
           :align: center

        Print out available attributes of DayNoise object

        >>> daynoise.__dict__.keys()
        dict_keys(['tr1', 'tr2', 'trZ', 'trP', 'window', 'overlap', 'key',
        'dt', 'npts', 'fs', 'year', 'julday', 'ncomp', 'tf_list', 'QC', 'av',
        'goodwins', 'f', 'power', 'cross', 'rotation'])

        """

        if not self.QC:
            print("Warning: processing daynoise object for " +
                  "QC_daily_spectra using default values")
            self.QC_daily_spectra()

        # Points in window
        ws = int(self.window/self.dt)

        # Number of points in step
        ss = int(self.window*(1.-self.overlap)/self.dt)

        ft1 = None
        ft2 = None
        ftZ = None
        ftP = None
        ftZ, f = utils.calculate_windowed_fft(self.trZ, ws, ss)
        if self.ncomp == 2 or self.ncomp == 4:
            ftP, f = utils.calculate_windowed_fft(self.trP, ws, ss)
        if self.ncomp == 3 or self.ncomp == 4:
            ft1, f = utils.calculate_windowed_fft(self.tr1, ws, ss)
            ft2, f = utils.calculate_windowed_fft(self.tr2, ws, ss)

        self.f = f

        # Extract good windows
        c11 = None
        c22 = None
        cZZ = None
        cPP = None
        cZZ = np.abs(
            np.mean(ftZ[self.goodwins, :]*np.conj(ftZ[self.goodwins, :]),
                    axis=0))[0:len(f)]
        if self.ncomp == 2 or self.ncomp == 4:
            cPP = np.abs(
                np.mean(ftP[self.goodwins, :]*np.conj(ftP[self.goodwins, :]),
                        axis=0))[0:len(f)]
        if self.ncomp == 3 or self.ncomp == 4:
            c11 = np.abs(
                np.mean(ft1[self.goodwins, :]*np.conj(ft1[self.goodwins, :]),
                        axis=0))[0:len(f)]
            c22 = np.abs(
                np.mean(ft2[self.goodwins, :]*np.conj(ft2[self.goodwins, :]),
                        axis=0))[0:len(f)]

        # Extract bad windows
        bc11 = None
        bc22 = None
        bcZZ = None
        bcPP = None
        if np.sum(~self.goodwins) > 0:
            bcZZ = np.abs(np.mean(
                ftZ[~self.goodwins, :]*np.conj(ftZ[~self.goodwins, :]),
                axis=0))[0:len(f)]
            if self.ncomp == 2 or self.ncomp == 4:
                bcPP = np.abs(np.mean(
                    ftP[~self.goodwins, :]*np.conj(ftP[~self.goodwins, :]),
                    axis=0))[0:len(f)]
            if self.ncomp == 3 or self.ncomp == 4:
                bc11 = np.abs(np.mean(
                    ft1[~self.goodwins, :]*np.conj(ft1[~self.goodwins, :]),
                    axis=0))[0:len(f)]
                bc22 = np.abs(np.mean(
                    ft2[~self.goodwins, :]*np.conj(ft2[~self.goodwins, :]),
                    axis=0))[0:len(f)]

        # Calculate mean of all good windows if component combinations exist
        c12 = None
        c1Z = None
        c2Z = None
        c1P = None
        c2P = None
        cZP = None
        if self.ncomp == 3 or self.ncomp == 4:
            c12 = np.mean(ft1[self.goodwins, :] *
                          np.conj(ft2[self.goodwins, :]), axis=0)[0:len(f)]
            c1Z = np.mean(ft1[self.goodwins, :] *
                          np.conj(ftZ[self.goodwins, :]), axis=0)[0:len(f)]
            c2Z = np.mean(ft2[self.goodwins, :] *
                          np.conj(ftZ[self.goodwins, :]), axis=0)[0:len(f)]
        if self.ncomp == 4:
            c1P = np.mean(ft1[self.goodwins, :] *
                          np.conj(ftP[self.goodwins, :]), axis=0)[0:len(f)]
            c2P = np.mean(ft2[self.goodwins, :] *
                          np.conj(ftP[self.goodwins, :]), axis=0)[0:len(f)]
        if self.ncomp == 2 or self.ncomp == 4:
            cZP = np.mean(ftZ[self.goodwins, :] *
                          np.conj(ftP[self.goodwins, :]), axis=0)[0:len(f)]

        # Store as attributes
        self.power = Power(c11, c22, cZZ, cPP)
        self.cross = Cross(c12, c1Z, c1P, c2Z, c2P, cZP)
        bad = Power(bc11, bc22, bcZZ, bcPP)

        if fig_average:
            plot = plotting.fig_average(f, self.power, bad, self.goodwins,
                                        self.ncomp, key=self.key)
            if save:
                fname = self.key + '.' + self.tkey + '.' + 'average.' + form
                if isinstance(save, Path):
                    fname = save / fname
                plot.savefig(
                    str(fname), dpi=300, bbox_inches='tight', format=form)
            else:
                plot.show()

        if calc_rotation and self.ncomp >= 3:
            cHH, cHZ, cHP, coh, ph, direc, tilt, coh_value, phase_value = \
                utils.calculate_tilt(
                    ft1, ft2, ftZ, ftP, f, self.goodwins)
            self.rotation = Rotation(
                cHH, cHZ, cHP, coh, ph, tilt, coh_value, phase_value, direc)

            if fig_coh_ph:
                plot = plotting.fig_coh_ph(coh, ph, direc)

                # Save or show figure
                if save:
                    fname = self.key + '.' + self.tkey + '.' + 'coh_ph.' + form
                    if isinstance(save, Path):
                        fname = save / fname
                    plot.savefig(
                        str(fname), dpi=300, bbox_inches='tight', format=form)
                else:
                    plot.show()

        else:
            self.rotation = Rotation()

        self.av = True

    def save(self, filename):
        """
        Method to save the object to file using `~Pickle`.

        Parameters
        ----------
        filename : str
            File name

        Examples
        --------

        Run demo through all methods

        >>> from obstools.atacr import DayNoise
        >>> daynoise = DayNoise('demo')
        Uploading demo data - March 04, 2012, station 7D.M08A
        >>> daynoise.QC_daily_spectra()
        >>> daynoise.average_daily_spectra()

        Save object

        >>> daynoise.save('daynoise_demo.pkl')

        Check that it has been saved

        >>> import glob
        >>> glob.glob("./daynoise_demo.pkl")
        ['./daynoise_demo.pkl']

        """

        if not self.av:
            print("Warning: saving before having calculated the average " +
                  "spectra")

        # Remove original traces to save disk space
        del self.tr1
        del self.tr2
        del self.trZ
        del self.trP

        file = open(str(filename), 'wb')
        pickle.dump(self, file)
        file.close()
