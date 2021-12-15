# import sys
# from scipy.signal import spectrogram, detrend
# from scipy.linalg import norm
# import matplotlib.pyplot as plt
import numpy as np
import pickle
from obspy.core import Trace  # , Stream, read
from obstools.atacr import utils  # , plotting
from pkg_resources import resource_filename
from pathlib import Path
np.seterr(all='ignore')
# np.set_printoptions(threshold=sys.maxsize)


class EventStream(object):
    """
    An EventStream object contains attributes that store station-event
    metadata and methods for applying the transfer functions to the various
    components and produce corrected/cleaned vertical components.

    Note
    ----
    An ``EventStream`` object is defined as the data
    (:class:`~obspy.core.Stream` object) are read from file or downloaded
    from an ``obspy`` Client. Based on the available components, a list of
    possible corrections is determined automatically.

    Attributes
    ----------
    sta : :class:`~stdb.StdbElement`
        An instance of an stdb object
    key : str
        Station key for current object
    sth : :class:`~obspy.core.Stream`
        Stream containing three-component seismic data (traces are empty if
        data are not available)
    stp : :class:`~obspy.core.Stream`
        Stream containing pressure data (trace is empty if data are not
        available)
    tstamp : str
        Time stamp for event
    evlat : float
        Latitude of seismic event
    evlon : float
        Longitude of seismic event
    evtime : :class:`~obspy.core.UTCDateTime`
        Origin time of seismic event
    window : float
        Length of time window in seconds
    fs : float
        Sampling frequency (in Hz)
    dt : float
        Sampling distance in seconds
    npts : int
        Number of points in time series
    ncomp : int
        Number of available components (either 2, 3 or 4)
    ev_list : Dict
        Dictionary of possible transfer functions given the available
        components. This is determined during initialization.
    correct : :class:`~obstools.atacr.classes.EventStream.CorrectDict`
        Container Dictionary for all possible corrections from the transfer
        functions. This is calculated from the method
        :func:`~obstools.atacr.classes.EventStream.correct_data`

    Examples
    --------

    Get demo earthquake data as EventStream object

    >>> from obstools.atacr import EventStream
    >>> evstream = EventStream('demo')
    Uploading demo earthquake data - March 09, 2012, station 7D.M08A
    >>> evstream.__dict__.keys()
    dict_keys(['sta', 'key', 'sth', 'stp', 'tstamp', 'evlat', 'evlon', 'evtime',
    'window', 'fs', 'dt', 'ncomp', 'ev_list'])

    Plot the raw traces

    >>> import obstools.atacr.plot as plot
    >>> plot.fig_event_raw(evstream, fmin=1./150., fmax=2.)

    .. figure:: ../obstools/examples/figures/Figure_11.png
       :align: center

    """

    def __init__(self, sta=None, sth=None, stp=None, tstamp=None, lat=None,
                 lon=None, time=None, window=None, sampling_rate=None,
                 ncomp=None, correct=False):

        if sta == 'demo' or sta == 'Demo':
            print("Uploading demo earthquake data - March 09, 2012, " +
                  "station 7D.M08A")
            exmpl_path = Path(resource_filename('obstools', 'examples'))
            fn = '2012.069.07.09.event.pkl'
            fn = exmpl_path / 'event' / fn
            evstream = pickle.load(open(fn, 'rb'))
            sta = evstream.sta
            key = evstream.key
            sth = evstream.sth
            stp = evstream.stp
            tstamp = evstream.tstamp
            lat = evstream.evlat
            lon = evstream.evlon
            time = evstream.evtime
            window = evstream.window
            sampling_rate = evstream.fs
            ncomp = evstream.ncomp
            correct = evstream.correct

        if any(value == None for value in [sta, sth, stp, tstamp, lat, lon,
                                           time, window, sampling_rate,
                                           ncomp]):
            raise(Exception(
                "Error: Initializing EventStream object with None values - " +
                "aborting"))

        self.sta = sta
        self.key = sta.network+'.'+sta.station
        self.sth = sth
        self.stp = stp
        self.tstamp = tstamp
        self.evlat = lat
        self.evlon = lon
        self.evtime = time
        self.window = window
        self.fs = sampling_rate
        self.dt = 1./sampling_rate
        self.ncomp = ncomp
        self.correct = False

        # Build list of available transfer functions for future use
        if self.ncomp == 2:
            self.ev_list = {'ZP': True, 'Z1': False, 'Z2-1': False,
                            'ZP-21': False, 'ZH': False, 'ZP-H': False, 'ZH-P': False}
        elif self.ncomp == 3:
            self.ev_list = {'ZP': False, 'Z1': True, 'Z2-1': True,
                            'ZP-21': False, 'ZH': True, 'ZP-H': False, 'ZH-P': False}
        else:
            self.ev_list = {'ZP': True, 'Z1': True, 'Z2-1': True,
                            'ZP-21': True, 'ZH': True, 'ZP-H': True, 'ZH-P': True}

    class CorrectDict(dict):

        def __init__(self):
            self = dict()

        def add(self, key, value):
            self[key] = value

    def correct_data(self, tfnoise, n_signif=0):
        """
        Method to apply transfer functions between multiple components (and
        component combinations) to produce corrected/cleaned vertical
        components.

        Parameters
        ----------
        tfnoise : :class:`~obstools.atacr.classes.TFNoise`
            Object that contains the noise transfer functions used in the
            correction
        n_signif (int): only apply transfer function correction if
            this number of neighboring coherencies are above the 95%
            significance level

        Attributes
        ----------
        correct : :class:`~obstools.atacr.classes.EventStream.CorrectDict`
            Container Dictionary for all possible corrections from the
            transfer functions

        Examples
        --------

        Let's carry through the correction of the vertical component for a
        single day of noise, say corresponding to the noise recorded on March
        04, 2012. In practice, the DayNoise object should correspond to the
        same day at that of the recorded earthquake to avoid bias in the
        correction.

        >>> from obstools.atacr import DayNoise, TFNoise, EventStream
        >>> daynoise = DayNoise('demo')
        Uploading demo data - March 04, 2012, station 7D.M08A
        >>> daynoise.QC_daily_spectra()
        >>> daynoise.average_daily_spectra()
        >>> tfnoise_day = TFNoise(daynoise)
        >>> tfnoise_day.transfer_func()
        >>> evstream = EventStream('demo')
        Uploading demo earthquake data - March 09, 2012, station 7D.M08A
        >>> evstream.correct_data(tfnoise_day)

        Plot the corrected traces

        >>> import obstools.atacr.plot as plot
        >>> plot.fig_event_corrected(evstream, tfnoise_day.tf_list)

        .. figure:: ../obstools/examples/figures/Figure_corrected_march04.png
           :align: center

        Carry out the same exercise but this time using a StaNoise object

        >>> from obstools.atacr import StaNoise, TFNoise, EventStream
        >>> stanoise = StaNoise('demo')
        Uploading demo data - March 01 to 04, 2012, station 7D.M08A
        >>> stanoise.QC_sta_spectra()
        >>> stanoise.average_sta_spectra()
        >>> tfnoise_sta = TFNoise(stanoise)
        >>> tfnoise_sta.transfer_func()
        >>> evstream = EventStream('demo')
        Uploading demo earthquake data - March 09, 2012, station 7D.M08A
        >>> evstream.correct_data(tfnoise_sta)

        Plot the corrected traces

        >>> import obstools.atacr.plot as plot
        >>> plot.fig_event_corrected(evstream, tfnoise_sta.tf_list)

        .. figure:: ../obstools/examples/figures/Figure_corrected_sta.png
           :align: center
        """
        if not tfnoise.transfunc:
            raise(
                Exception("Error: Object TFNoise has no transfunc " +
                          "attribute - aborting"))
        if n_signif:
            if not tfnoise.coher:
                raise(
                    Exception("Error: Object TFNoise has no coher " +
                              "attribute - aborting"))
            coher = tfnoise.coher
            n_wins = tfnoise.n_wins
            

        correct = self.CorrectDict()

        # Extract list and transfer functions available
        tf_list = tfnoise.tf_list
        transfunc = tfnoise.transfunc

        # Points in window
        ws = int(self.window/self.dt)

        # Extract traces
        trZ, tr1, tr2, trP = Trace(), Trace(), Trace(), Trace()
        trZ = self.sth.select(component='Z')[0]
        if self.ncomp == 2 or self.ncomp == 4:
            trP = self.stp[0]
        if self.ncomp == 3 or self.ncomp == 4:
            tr1 = self.sth.select(component='1')[0]
            tr2 = self.sth.select(component='2')[0]

        # Get Fourier spectra
        ft1, ft2, ftZ, ftP = None, None, None, None
        ftZ, f = utils.calculate_windowed_fft(trZ, ws, hann=False)
        if self.ncomp == 2 or self.ncomp == 4:
            ftP, f = utils.calculate_windowed_fft(trP, ws, hann=False)
        if self.ncomp == 3 or self.ncomp == 4:
            ft1, f = utils.calculate_windowed_fft(tr1, ws, hann=False)
            ft2, f = utils.calculate_windowed_fft(tr2, ws, hann=False)

        if not np.allclose(f, tfnoise.f):
            raise(Exception('Frequency axes are different: ', f, tfnoise.f))

        # set up self._fTF() calls
        self._tfnoise = tfnoise
        self._nF = len(f)
        self._n_signif = n_signif

        for key, value in tf_list.items():
            if not value:
                continue
            if key == 'ZP' and self.ev_list[key] and tf_list[key]:
                fTF_ZP = self._fTF(key, 'TF_ZP')
                corrspec = ftZ - fTF_ZP*ftP
            elif key == 'Z1' and self.ev_list[key] and tf_list[key]:
                fTF_Z1 = self._fTF(key, 'TF_Z1')
                corrspec = ftZ - fTF_Z1*ft1
            elif key == 'Z2-1' and self.ev_list[key] and tf_list[key]:
                fTF_Z1 = self._fTF('Z1', 'TF_Z1')
                fTF_21 = self._fTF(key, 'TF_21')
                fTF_Z2_1 = self._fTF(key, 'TF_Z2-1')
                corrspec = ftZ - fTF_Z1*ft1 - (ft2 - ft1*fTF_21)*fTF_Z2_1
            elif key == 'ZP-21' and self.ev_list[key] and tf_list[key]:
                fTF_Z1 = self._fTF(key, 'TF_Z1')
                fTF_21 = self._fTF(key, 'TF_21')
                fTF_Z2_1 = self._fTF(key, 'TF_Z2-1')
                fTF_P1 = self._fTF(key, 'TF_P1')
                fTF_P2_1 = self._fTF(key, 'TF_P2-1')
                fTF_ZP_21 = self._fTF(key, 'TF_ZP-21')
                corrspec = (ftZ
                            - fTF_Z1*ft1
                            - (ft2 - ft1*fTF_21)*fTF_Z2_1
                            - (ftP - ft1*fTF_P1 - (ft2 - ft1*fTF_21)*fTF_P2_1)*fTF_ZP_21)
            elif key == 'ZH' and self.ev_list[key] and tf_list[key]:
                ftH = utils.rotate_dir(ft1, ft2, tfnoise.tilt)
                fTF_ZH = self._fTF(key, 'TF_ZH')
                corrspec = ftZ - fTF_ZH*ftH
            elif key == 'ZP-H' and self.ev_list[key] and tf_list[key]:
                ftH = utils.rotate_dir(ft1, ft2, tfnoise.tilt)
                fTF_ZH = self._fTF('ZH', 'TF_ZH')
                fTF_PH = self._fTF('ZP-H', 'TF_PH')
                fTF_ZP_H = self._fTF('ZP-H', 'TF_ZP-H')
                corrspec = ftZ - fTF_ZH*ftH - (ftP - ftH*fTF_PH)*fTF_ZP_H
            elif key == 'ZH-P' and self.ev_list[key] and tf_list[key]:
                ftH = utils.rotate_dir(ft1, ft2, tfnoise.tilt)
                fTF_ZP = self._fTF('ZP', 'TF_ZP')
                fTF_HP = self._fTF('ZH-P', 'TF_HP')
                fTF_ZH_P = self._fTF('ZH-P', 'TF_ZH-P')
                corrspec = ftZ - ftP*fTF_ZP - (ftH - ftP*fTF_HP)*fTF_ZH_P
            else:
                continue

            corrtime = np.real(np.fft.ifft(corrspec))[0:ws]
            correct.add(key, corrtime)
            # correct.add(key+'_spec', corrspec)  # WCC added to get spectra as well
        self.correct = correct

    def _fTF(self, key1, key2):
        """
        Calculate Fourier Transfer Function (WCC)
        
        Args:
            key1 (str): first key of transfer function / coherency
            key2 (str): second key of transfer function / coherency
        """
        TF = self._tfnoise.transfunc[key1][key2]
        if self._n_signif:
            coh = self._tfnoise.coher[key1][key2]
            TF[~self._good_cohers(coh, self._tfnoise.n_wins, self._n_signif)] = 0
        return np.hstack((TF, np.conj(TF[::-1][1:self._nF-1])))

    def _good_cohers(self, coher, n_wins, n_signif=3):
        """
        Return boolean array indicating where coherency is significant (WCC)
        
        Args:
            coher (:class:`~numpy.ndarray`): coherency
            n_signif: number of neighboring coherencies that need to be above the
                     95% significance level
            n_wins (int): number of windows used to calcualate tf and coher
        Returns:
            (:class:`~numpy.ndarray`): boolean array of significant coherences

        >>> from obstools.atacr import EventStream
        >>> from numpy import arange
        >>> cohers = arange(0, 1, 0.05)
        >>> cohers[13] = 0
        >>> EventStream._good_cohers(cohers, 40, 1)
        array([False, False, False, False, False,  True,  True,  True,  True,
        True,  True,  True,  True, False,  True,  True,  True,  True,
        True,  True], dtype=bool)
        >>> EventStream._good_cohers(cohers, 40, 3)
        array([False, False, False, False, False, False, False,  True,  True,
        True,  True,  True,  True, False, False, False,  True,  True,
        True,  True], dtype=bool)
        >>> EventStream._good_cohers(cohers, 40, 5)
        array([False, False, False, False, False, False, False,  True,  True,
        True,  True, False, False, False, False, False,  True,  True,
        True,  True], dtype=bool)
        >>> cohers[~EventStream._good_cohers(cohers, 40, 5)] = 0
        >>> cohers
        array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.35,  0.4 ,
        0.45,  0.5 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.8 ,  0.85,
        0.9 ,  0.95])
        """
        if n_signif == 0:
            return np.ones(size(coher))
        signif_level = np.sqrt(2/n_wins)  # 95% significancy level
        good_coher = np.abs(coher) > signif_level
        if n_signif > 1:
            # Propagate bad coherences n_signif-1 to the right
            good_coher_orig = good_coher.copy()
            for i in np.arange(1, n_signif):
                good_coher &= np.concatenate((good_coher_orig[:i],
                                              np.roll(good_coher_orig, i)[i:]))
            # Shift good_coher back to the original indices
            if n_signif >= 3:
                left_shift = int(n_signif/2)
                good_coher = np.concatenate((np.roll(good_coher, -left_shift)[:-left_shift],
                                             good_coher[-left_shift:]))
        return good_coher

    def save(self, filename):
        """
        Method to save the object to file using `~Pickle`.

        Parameters
        ----------
        filename : str
            File name

        Examples
        --------

        Following from the example outlined in method
        :func:`~obstools.atacr.classes.EventStream.correct_data`, we simply
        save the final object

        >>> evstream.save('evstream_demo.pkl')

        Check that object has been saved

        >>> import glob
        >>> glob.glob("./evstream_demo.pkl")
        ['./evstream_demo.pkl']

        """

        if not self.correct:
            print("Warning: saving EventStream object before having done " +
                  "the corrections")

        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()
