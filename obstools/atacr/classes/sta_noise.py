# Copyright 2019 Pascal Audet & Helen Janiszewski
#
# This file is part of OBStools.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# import sys
from scipy.signal import detrend  # , spectrogram
from scipy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from obspy.core import read  # , Stream, Trace
from obstools.atacr import utils, plotting
from pkg_resources import resource_filename
from pathlib import Path

from .containers import Power, Cross, Rotation
from .day_noise import DayNoise

np.seterr(all='ignore')
# np.set_printoptions(threshold=sys.maxsize)


class StaNoise(object):
    """
    A StaNoise object contains attributes that associate
    three-component raw (or deconvolved) traces, metadata information
    and window parameters.

    Note
    ----
    The object is initially a container for
    :class:`~obstools.atacr.classes.DayNoise` objects. Once the StaNoise
    object is initialized (using the method `init()` or by calling the
    `QC_sta_spectra` method), each individual spectral quantity is unpacked
    as an object attribute and the original `DayNoise` objects are removed
    from memory. In addition, all spectral quantities associated with the
    original `DayNoise` objects (now stored as attributes) are discarded as
    the object is saved to disk and new container objects are defined and
    saved.

    Attributes
    ----------
    daylist : list
        A list of :class:`~obstools.atacr.classes.DayNoise` objects to process
        and produce a station average
    initialized : bool
        Whether or not the object has been initialized - `False` unless one
        of the methods have been called. When `True`, the `daylist` attribute
        is deleted from memory

    Examples
    --------

    Initialize empty object

    >>> from obstools.atacr import StaNoise
    >>> stanoise = StaNoise()

    Initialize with DayNoise object

    >>> from obstools.atacr import DayNoise
    >>> daynoise = DayNoise('demo')
    Uploading demo data - March 04, 2012, station 7D.M08A
    >>> stanoise = StaNoise(daylist=[daynoise])

    Add or append DayNoise object to StaNoise

    >>> stanoise = StaNoise()
    >>> stanoise += daynoise

    >>> stanoise = StaNoise()
    >>> stanoise.append(daynoise)

    Import demo noise data with 4 DayNoise objects

    >>> from obstools.atacr import StaNoise
    >>> stanoise = StaNoise('demo')
    Uploading demo data - March 01 to 04, 2012, station 7D.M08A
    >>> stanoise.daylist
    [<obstools.atacr.classes.DayNoise at 0x11e3ce8d0>,
     <obstools.atacr.classes.DayNoise at 0x121c7ae10>,
     <obstools.atacr.classes.DayNoise at 0x121ca5940>,
     <obstools.atacr.classes.DayNoise at 0x121e7dd30>]
     >>> sta.initialized
     False

    """

    def __init__(self, daylist=None):

        def _load_dn(day):
            exmpl_path = Path(resource_filename('obstools', 'examples'))
            fn = '2012.'+day+'*.SAC'
            fn = exmpl_path / 'data' / fn
            st = read(str(fn))
            tr1 = st.select(component='1')[0]
            tr2 = st.select(component='2')[0]
            trZ = st.select(component='Z')[0]
            trP = st.select(component='P')[0]
            window = 7200.
            overlap = 0.3
            key = '7D.M08A'
            return DayNoise(tr1, tr2, trZ, trP, window, overlap, key)

        self.daylist = []
        self.initialized = False
        self.QC = False
        self.av = False

        if isinstance(daylist, DayNoise):
            daylist = [daylist]
        elif daylist == 'demo' or daylist == 'Demo':
            print("Uploading demo data - March 01 to 04, 2012, station " +
                  "7D.M08A")
            self.daylist = [_load_dn('061'), _load_dn(
                '062'), _load_dn('063'), _load_dn('064')]
        if not daylist == 'demo' and daylist:
            self.daylist.extend(daylist)

    def __add__(self, other):

        if isinstance(other, DayNoise):
            other = StaNoise([other])
        if not isinstance(other, StaNoise):
            raise TypeError
        daylist = self.daylist + other.daylist
        return self.__class__(daylist=daylist)

    def __iter__(self):

        return List(self.daylist).__iter__()

    def append(self, daynoise):

        if isinstance(daynoise, DayNoise):
            self.daylist.append(daynoise)
        else:
            msg = 'Append only supports a single DayNoise object as argument'
            raise TypeError(msg)
        return self

    def extend(self, daynoise_list):

        if isinstance(daynoise_list, list):
            for _i in daynoise_list:
                # Make sure each item in the list is a Grid object.
                if not isinstance(_i, DayNoise):
                    msg = 'Extend only accepts a list of Daynoise objects.'
                    raise TypeError(msg)
            self.daylist.extend(daynoise_list)
        elif isinstance(daynoise_list, StaNoise):
            self.daylist.extend(daynoise_list.daylist)
        else:
            msg = 'Extend only supports a list of DayNoise objects as ' +\
                'argument.'
            raise TypeError(msg)
        return self

    def init(self):
        """
        Method to initialize the `StaNoise` object. This method is used to
        unpack the spectral quantities from the original
        :class:`~obstools.atacr.classes.DayNoise` objects and allow the
        methods to proceed. The original
        :class:`~obstools.atacr.classes.DayNoise` objects are deleted from
        memory during this process.

        Note
        ----
        If the original :class:`~obstools.atacr.classes.DayNoise` objects
        have not been processed using their QC and averaging methods, these
        will be called first before unpacking into the object attributes.

        Attributes
        ----------
        f : :class:`~numpy.ndarray`
            Frequency axis for corresponding time sampling parameters
        nwins : int
            Number of good windows from the
            :class:`~obstools.atacr.classes.DayNoise` object
        key : str
            Station key for current object
        ncomp : int
            Number of available components (either 2, 3 or 4)
        tf_list : Dict
            Dictionary of possible transfer functions given the available
            components.
        c11 : `numpy.ndarray`
            Power spectra for component `H1`. Other identical attributes
            are available for
            the power, cross and rotated spectra: [11, 12, 1Z, 1P, 22, 2Z,
            2P, ZZ, ZP, PP, HH, HZ, HP]
        direc : `numpy.ndarray`
            Array of azimuths used in determining the tilt direction
        tilt : float
            Tilt direction from maximum coherence between rotated `H1` and
            `HZ` components
        QC : bool
            Whether or not the method
            :func:`~obstools.atacr.classes.StaNoise.QC_sta_spectra` has
            been called.
        av : bool
            Whether or not the method
            :func:`~obstools.atacr.classes.StaNoise.average_sta_spectra` has
            been called.

        Examples
        --------

        Initialize demo data

        >>> from obstools.atacr import StaNoise
        >>> stanoise = StaNoise('demo')
        Uploading demo data - March 01 to 04, 2012, station 7D.M08A
        >>> stanoise.init()

        Check that `daylist` attribute has been deleted

        >>> stanoise.daylist
        ---------------------------------------------------------------------------
        AttributeError                            Traceback (most recent call last)
        <ipython-input-4-a292a91450a9> in <module>
        ----> 1 stanoise.daylist
        AttributeError: 'StaNoise' object has no attribute 'daylist'
        >>> stanoise.__dict__.keys()
        dict_keys(['initialized', 'c11', 'c22', 'cZZ', 'cPP', 'c12', 'c1Z', 'c1P',
        'c2Z', 'c2P', 'cZP', 'cHH', 'cHZ', 'cHP', 'direc', 'tilt', 'f', 'nwins',
        'ncomp', 'key', 'tf_list', 'QC', 'av'])

        """

        # First, check that the StaNoise object contains at least two
        # DayNoise objects
        if len(self.daylist) < 2:
            raise(Exception(
                "StaNoise requires at least two DayNoise objects to execute " +
                "its methods"))

        for dn in self.daylist:
            if not dn.QC:
                dn.QC_daily_spectra()
            if not dn.av:
                dn.average_daily_spectra()

        # Then unpack the DayNoise objects
        self.c11 = np.array([dn.power.c11 for dn in self.daylist]).T
        self.c22 = np.array([dn.power.c22 for dn in self.daylist]).T
        self.cZZ = np.array([dn.power.cZZ for dn in self.daylist]).T
        self.cPP = np.array([dn.power.cPP for dn in self.daylist]).T
        self.c12 = np.array([dn.cross.c12 for dn in self.daylist]).T
        self.c1Z = np.array([dn.cross.c1Z for dn in self.daylist]).T
        self.c1P = np.array([dn.cross.c1P for dn in self.daylist]).T
        self.c2Z = np.array([dn.cross.c2Z for dn in self.daylist]).T
        self.c2P = np.array([dn.cross.c2P for dn in self.daylist]).T
        self.cZP = np.array([dn.cross.cZP for dn in self.daylist]).T
        self.cHH = np.array([dn.rotation.cHH for dn in self.daylist]).T
        self.cHZ = np.array([dn.rotation.cHZ for dn in self.daylist]).T
        self.cHP = np.array([dn.rotation.cHP for dn in self.daylist]).T
        self.direc = self.daylist[0].rotation.direc
        self.tilt = self.daylist[0].rotation.tilt
        self.f = self.daylist[0].f
        self.nwins = np.array([np.sum(dn.goodwins) for dn in self.daylist])
        self.ncomp = np.min([dn.ncomp for dn in self.daylist])
        self.key = self.daylist[0].key

        # Build list of available transfer functions for future use
        if self.ncomp == 2:
            self.tf_list = {'ZP': True, 'Z1': False, 'Z2-1': False,
                            'ZP-21': False, 'ZH': False, 'ZP-H': False}
        elif self.ncomp == 3:
            self.tf_list = {'ZP': False, 'Z1': True, 'Z2-1': True,
                            'ZP-21': False, 'ZH': False, 'ZP-H': False}
        else:
            self.tf_list = {'ZP': True, 'Z1': True, 'Z2-1': True,
                            'ZP-21': True, 'ZH': False, 'ZP-H': False}

        self.initialized = True
        self.QC = False
        self.av = False

        # Remove DayNoise objects from memory
        del self.daylist

    def QC_sta_spectra(self, pd=[0.004, 0.2], tol=2.0, alpha=0.05,
                       fig_QC=False, debug=False, save=False, form='png'):
        """
        Method to determine the days (for given time window) for which the
        spectra are anomalous and should be discarded in the calculation of
        the long-term transfer functions.

        Parameters
        ----------
        pd : list
            Frequency corners of passband for calculating the spectra
        tol : float
            Tolerance threshold. If spectrum > std*tol, window is flagged as
            bad
        alpha : float
            Confidence interval for f-test
        fig_QC : boolean
            Whether or not to produce a figure showing the results of the
            quality control
        debug : boolean
            Whether or not to plot intermediate steps in the QC procedure for
            debugging

        Attributes
        ----------
        gooddays : list
            List of booleans representing whether a day is good (True) or not
            (False)

        Examples
        --------
        Import demo data, call method and generate final figure

        >>> obstools.atacr import StaNoise
        >>> stanoise = StaNoise('demo')
        Uploading demo data - March 01 to 04, 2012, station 7D.M08A
        >>> stanoise.QC_sta_spectra(fig_QC=True)
        >>> stanoise.QC
        True

        """

        if self.initialized:
            raise(Exception("Object has been initialized already - " +
                            "list of DayNoise objects has been lost and " +
                            "method cannot proceed"))
        else:
            self.init()

        # Select bandpass frequencies
        ff = (self.f > pd[0]) & (self.f < pd[1])

        # Smooth out the log of the PSDs
        sl_cZZ = None
        sl_c11 = None
        sl_c22 = None
        sl_cPP = None
        sl_cZZ = utils.smooth(np.log(self.cZZ), 50, axis=0)
        if self.ncomp == 2 or self.ncomp == 4:
            sl_cPP = utils.smooth(np.log(self.cPP), 50, axis=0)
        if self.ncomp == 3 or self.ncomp == 4:
            sl_c11 = utils.smooth(np.log(self.c11), 50, axis=0)
            sl_c22 = utils.smooth(np.log(self.c22), 50, axis=0)

        # Remove mean of the log PSDs
        dsl_cZZ = sl_cZZ[ff, :] - np.mean(sl_cZZ[ff, :], axis=0)
        if self.ncomp == 2:
            dsl_cPP = sl_cPP[ff, :] - np.mean(sl_cPP[ff, :], axis=0)
            dsls = [dsl_cZZ, dsl_cPP]
        elif self.ncomp == 3:
            dsl_c11 = sl_c11[ff, :] - np.mean(sl_c11[ff, :], axis=0)
            dsl_c22 = sl_c22[ff, :] - np.mean(sl_c22[ff, :], axis=0)
            dsls = [dsl_c11, dsl_c22, dsl_cZZ]
        else:
            dsl_c11 = sl_c11[ff, :] - np.mean(sl_c11[ff, :], axis=0)
            dsl_c22 = sl_c22[ff, :] - np.mean(sl_c22[ff, :], axis=0)
            dsl_cPP = sl_cPP[ff, :] - np.mean(sl_cPP[ff, :], axis=0)
            dsls = [dsl_c11, dsl_c22, dsl_cZZ, dsl_cPP]

        plt.figure(2)
        if self.ncomp == 2:
            self._plot_sl_psds(self.f, (sl_cZZ, sl_cPP), ('g', 'k'))
        elif self.ncomp == 3:
            self._plot_sl_psds(self.f, (sl_c11, sl_c22, sl_cZZ),
                               ('r', 'b', 'g'))
        else:
            self._plot_sl_psds(self.f, (sl_c11, sl_c22, sl_cZZ, sl_cPP),
                               ('r', 'b', 'g', 'k'))
        if debug:
            plt.show()

        # Cycle through to kill high-std-norm windows
        moveon = False
        gooddays = np.repeat([True], self.cZZ.shape[1])
        indwin = np.argwhere(gooddays == True)

        while moveon == False:
            ubernorm = np.empty((self.ncomp, np.sum(gooddays)))
            for ind_u, dsl in enumerate(dsls):
                normvar = np.zeros(np.sum(gooddays))
                for ii, tmp in enumerate(indwin):
                    ind = np.copy(indwin)
                    ind = np.delete(ind, ii)
                    normvar[ii] = norm(np.std(dsl[:, ind], axis=1), ord=2)
                ubernorm[ind_u, :] = np.median(normvar) - normvar

            penalty = np.sum(ubernorm, axis=0)

            plt.figure(4)
            for i in range(self.ncomp):
                plt.plot(range(0, np.sum(gooddays)), detrend(
                    ubernorm, type='constant')[i], 'o-')
            if debug:
                plt.show()
            else:
                plt.close(4)
            plt.figure(5)
            plt.plot(range(0, np.sum(gooddays)),
                     np.sum(ubernorm, axis=0), 'o-')
            if debug:
                plt.show()
            else:
                plt.close(5)

            kill = penalty > tol*np.std(penalty)
            if np.sum(kill) == 0:
                self.gooddays = gooddays
                self.QC = True
                moveon = True

            trypenalty = penalty[np.argwhere(kill == False)].T[0]

            if utils.ftest(penalty, 1, trypenalty, 1) < alpha:
                gooddays[indwin[kill == True]] = False
                indwin = np.argwhere(gooddays == True)
                moveon = False
            else:
                moveon = True

        self.gooddays = gooddays
        self.QC = True

        if fig_QC:
            power = Power(sl_c11, sl_c22, sl_cZZ, sl_cPP)
            plot = plotting.fig_QC(self.f, power, gooddays,
                                   self.ncomp, key=self.key)
            if save:
                fname = self.key + '.' + 'QC.' + form
                if isinstance(save, Path):
                    fname = save / fname
                plot.savefig(
                    str(fname), dpi=300, bbox_inches='tight', format=form)
            else:
                plot.show()

    @staticmethod
    def _plot_sl_psds(f, sl_psds, colors):
        n_psds = len(sl_psds)
        if not len(colors) == n_psds:
            ValueError('sl_psds and colors are not the same length')
        for sl_psd, color, i in zip(sl_psds, colors, range(n_psds)):
            plt.subplot(n_psds, 1, i+1)
            plt.semilogx(f, sl_psd, color, lw=0.5)
        plt.tight_layout()

    def plot_cohers(self):
        if self.ncomp==2:
            fig, ax = plt.subplots(1, 1)
            self._plot_coher(ax, 'Z-1')
        elif self.ncomp == 3:
            fig, axs = plt.subplots(2, 2)
            self._plot_coher(axs[0, 0], 'Z-1')
            self._plot_coher(axs[0, 1], 'Z-2')
            self._plot_coher(axs[1, 1], '1-2')            
        else:
            fig, axs = plt.subplots(3, 3)
            self._plot_coher(axs[0, 0], 'Z-1')
            self._plot_coher(axs[0, 1], 'Z-2')
            self._plot_coher(axs[0, 2], 'Z-P')
            self._plot_coher(axs[1, 1], '1-2')
            self._plot_coher(axs[1, 2], '1-P')
            self._plot_coher(axs[2, 2], '2-P')

    def _plot_coher(self, ax, which):
        if which == 'Z-P':
            coh = np.abs(self.cZP)**2/(self.cZZ*self.cPP)
            pha = np.angle(self.cZP)
        elif which == 'Z-1':
            coh = np.abs(self.c1Z)**2/(self.cZZ*self.c11)
            pha = np.angle(self.c1Z)
        elif which == 'Z-2':
            coh = np.abs(self.c2Z)**2/(self.cZZ*self.c22)
            pha = np.angle(self.c2Z)
        elif which == '1-2':
            coh = np.abs(self.c12)**2/(self.c11*self.c22)
            pha = np.angle(self.c12)
        elif which == '1-P':
            coh = np.abs(self.c1P)**2/(self.c11*self.cPP)
            pha = np.angle(self.c1P)
        elif which == '2-P':
            coh = np.abs(self.c2P)**2/(self.c22*self.cPP)
            pha = np.angle(self.c2P)
        else:
            ValueError(f'Unkown component combination: "{which}"')
        ax.semilogx(self.f, coh)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(f'{which} Coherence')
        ax.set_ylim((0, 1))

    def average_sta_spectra(self, fig_average=False, save=False, form='png'):
        r"""
        Method to average the daily station spectra for good windows.

        Parameters
        ----------
        fig_average : boolean
            Whether or not to produce a figure showing the average daily
            spectra

        Attributes
        ----------
        power : :class:`~obstools.atacr.classes.Power`
            Container for the Power spectra
        cross : :class:`~obstools.atacr.classes.Cross`
            Container for the Cross power spectra
        rotation : :class:`~obstools.atacr.classes.Cross`, optional
            Container for the Rotated power and cross spectra

        Examples
        --------
        Average daily spectra for good days using default values and produce
        final figure

        >>> obstools.atacr import StaNoise
        >>> stanoise = StaNoise('demo')
        Uploading demo data - March 01 to 04, 2012, station 7D.M08A
        >>> stanoise.QC_sta_spectra()
        >>> stanoise.average_sta_spectra()

        """

        if not self.QC:
            print("Warning: processing StaNoise object for QC_sta_spectra "
                  "using default values")
            self.QC_sta_spectra()

        # Power spectra
        c11 = None
        c22 = None
        cZZ = None
        cPP = None
        cZZ = np.sum(self.cZZ[:, self.gooddays]*self.nwins[self.gooddays],
                     axis=1) / np.sum(self.nwins[self.gooddays])
        if self.ncomp == 2 or self.ncomp == 4:
            cPP = np.sum(self.cPP[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
        if self.ncomp == 3 or self.ncomp == 4:
            c11 = np.sum(self.c11[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
            c22 = np.sum(self.c22[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])

        # Bad days - for plotting
        bc11 = None
        bc22 = None
        bcZZ = None
        bcPP = None
        if np.sum(~self.gooddays) > 0:
            bcZZ = np.sum(
                self.cZZ[:, ~self.gooddays]*self.nwins[~self.gooddays],
                axis=1) / np.sum(self.nwins[~self.gooddays])
            if self.ncomp == 2 or self.ncomp == 4:
                bcPP = np.sum(
                    self.cPP[:, ~self.gooddays]*self.nwins[~self.gooddays],
                    axis=1) / np.sum(self.nwins[~self.gooddays])
            if self.ncomp == 3 or self.ncomp == 4:
                bc11 = np.sum(
                    self.c11[:, ~self.gooddays]*self.nwins[~self.gooddays],
                    axis=1) / np.sum(self.nwins[~self.gooddays])
                bc22 = np.sum(
                    self.c22[:, ~self.gooddays]*self.nwins[~self.gooddays],
                    axis=1) / np.sum(self.nwins[~self.gooddays])

        # Cross spectra
        c12 = None
        c1Z = None
        c2Z = None
        c1P = None
        c2P = None
        cZP = None
        if self.ncomp == 3 or self.ncomp == 4:
            c12 = np.sum(self.c12[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
            c1Z = np.sum(self.c1Z[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
            c2Z = np.sum(self.c2Z[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
        if self.ncomp == 4:
            c1P = np.sum(self.c1P[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
            c2P = np.sum(self.c2P[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
        if self.ncomp == 2 or self.ncomp == 4:
            cZP = np.sum(self.cZP[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])

        # Rotated component
        cHH = None
        cHZ = None
        cHP = None
        if self.ncomp == 3 or self.ncomp == 4:
            cHH = np.sum(self.cHH[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
            cHZ = np.sum(self.cHZ[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])
        if self.ncomp == 4:
            cHP = np.sum(self.cHP[:, self.gooddays]*self.nwins[self.gooddays],
                         axis=1) / np.sum(self.nwins[self.gooddays])

        self.power = Power(c11, c22, cZZ, cPP)
        self.cross = Cross(c12, c1Z, c1P, c2Z, c2P, cZP)
        self.rotation = Rotation(cHH, cHZ, cHP)
        bad = Power(bc11, bc22, bcZZ, bcPP)

        if fig_average:
            plot = plotting.fig_average(self.f, self.power, bad, self.gooddays,
                                        self.ncomp, key=self.key)
            if save:
                fname = self.key + '.' + 'average.' + form
                if isinstance(save, Path):
                    fname = save / fname
                plot.savefig(
                    str(fname), dpi=300, bbox_inches='tight', format=form)
            else:
                plot.show()

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

        >>> from obstools.atacr import StaNoise
        >>> stanoise = StaNoise('demo')
        Uploading demo data - March 01 to 04, 2012, station 7D.M08A
        >>> stanoise.QC_sta_spectra()
        >>> stanoise.average_sta_spectra()

        Save object

        >>> stanoise.save('stanoise_demo.pkl')

        Check that it has been saved

        >>> import glob
        >>> glob.glob("./stanoise_demo.pkl")
        ['./stanoise_demo.pkl']

        """

        if not self.av:
            print("Warning: saving before having calculated the average " +
                  "spectra")

        # Remove traces to save disk space
        del self.c11
        del self.c22
        del self.cZZ
        del self.cPP
        del self.c12
        del self.c1Z
        del self.c1P
        del self.c2Z
        del self.c2P
        del self.cZP

        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()
