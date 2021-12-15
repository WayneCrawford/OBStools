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
import pickle

import numpy as np
from matplotlib import pyplot as plt

from obstools.atacr import utils  # , plotting
from .day_noise import DayNoise
from .sta_noise import StaNoise

np.seterr(all='ignore')
# np.set_printoptions(threshold=sys.maxsize)


class TFNoise(object):
    """
    A TFNoise object contains attributes that store the transfer function
    information from multiple components (and component combinations).

    Note
    ----
    The object is initialized with either a processed
    :class:`~obstools.atacr.classes.DayNoise` or
    :class:`~obstools.atacr.classes.StaNoise` object. Each individual
    spectral quantity is unpacked as an object attribute, but all of them
    are discarded as the object is saved to disk and new container objects
    are defined and saved.

    Attributes
    ----------
    f : :class:`~numpy.ndarray`
        Frequency axis for corresponding time sampling parameters
    c11 : `numpy.ndarray`
        Power spectra for component `H1`. Other identical attributes are
        available for the power, cross and rotated spectra:
        [11, 12, 1Z, 1P, 22, 2Z, 2P, ZZ, ZP, PP, HH, HZ, HP]
    tilt : float
        Tilt direction from maximum coherence between rotated `H1` and
        `HZ` components
    tf_list : Dict
        Dictionary of possible transfer functions given the available
        components.

    Examples
    --------

    Initialize a TFNoise object with a DayNoise object. The DayNoise
    object must be processed for QC and averaging, otherwise the TFNoise
    object will not initialize.

    >>> from obstools.atacr import DayNoise, TFNoise
    >>> daynoise = DayNoise('demo')
    Uploading demo data - March 04, 2012, station 7D.M08A
    >>> tfnoise = TFNoise(daynoise)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/Users/pascalaudet/Softwares/Python/Projects/dev/OBStools/obstools/atacr/classes.py", line 1215, in __init__
    Exception: Error: Noise object has not been processed (QC and averaging) - aborting

    Now re-initialized with a processed DayNoise object

    >>> from obstools.atacr import DayNoise, TFNoise
    >>> daynoise = DayNoise('demo')
    Uploading demo data - March 04, 2012, station 7D.M08A
    >>> daynoise.QC_daily_spectra()
    >>> daynoise.average_daily_spectra()
    >>> tfnoise = TFNoise(daynoise)

    Initialize a TFNoise object with a processed StaNoise object

    >>> from obstools.atacr import StaNoise, TFNoise
    >>> stanoise = StaNoise('demo')
    Uploading demo data - March 01 to 04, 2012, station 7D.M08A
    >>> stanoise.QC_sta_spectra()
    >>> stanoise.average_sta_spectra()
    >>> tfnoise = TFNoise(stanoise)
    """
    def __init__(self, objnoise=None):

        if (not objnoise and not isinstance(objnoise, DayNoise) and
                not isinstance(objnoise, StaNoise)):
            raise TypeError("Error: A TFNoise object must be initialized with"
                            " only one of type DayNoise or StaNoise object")

        if not objnoise.av:
            raise(Exception("Error: Noise object has not been processed (QC "
                            "and averaging) - aborting"))

        self.f = objnoise.f
        self.c11 = objnoise.power.c11
        self.c22 = objnoise.power.c22
        self.cZZ = objnoise.power.cZZ
        self.cPP = objnoise.power.cPP
        self.cHH = objnoise.rotation.cHH
        self.cHZ = objnoise.rotation.cHZ
        self.cHP = objnoise.rotation.cHP
        self.c12 = objnoise.cross.c12
        self.c1Z = objnoise.cross.c1Z
        self.c1P = objnoise.cross.c1P
        self.c2Z = objnoise.cross.c2Z
        self.c2P = objnoise.cross.c2P
        self.cZP = objnoise.cross.cZP
        self.tilt = objnoise.rotation.tilt
        self.tf_list = objnoise.tf_list
        self.transfunc = None
        self.coher = None
        if hasattr(objnoise, 'nwins'):
            self.n_wins = objnoise.nwins
        else:
            self.n_wins = np.sum(objnoise.goodwins)

    class TfDict(dict):

        def __init__(self):
            self = dict()

        def add(self, key, value):
            self[key] = value

    def transfer_func(self):
        """
        Method to calculate transfer functions between multiple
        components (and component combinations) from the averaged
        (daily or station-averaged) noise spectra.

        Attributes
        ----------
        transfunc : :class:`~obstools.atacr.classes.TFNoise.TfDict`
            Container Dictionary for all possible transfer functions

        Examples
        --------

        Calculate transfer functions for a DayNoise object

        >>> from obstools.atacr import DayNoise, TFNoise
        >>> daynoise = DayNoise('demo')
        Uploading demo data - March 04, 2012, station 7D.M08A
        >>> daynoise.QC_daily_spectra()
        >>> daynoise.average_daily_spectra()
        >>> tfnoise = TFNoise(daynoise)
        >>> tfnoise.transfer_func()
        >>> tfnoise.transfunc.keys()
        dict_keys(['ZP', 'Z1', 'Z2-1', 'ZP-21', 'ZH', 'ZP-H'])

        Calculate transfer functions for a StaNoise object

        >>> from obstools.atacr import StaNoise, TFNoise
        >>> stanoise = StaNoise('demo')
        Uploading demo data - March 01 to 04, 2012, station 7D.M08A
        >>> stanoise.QC_sta_spectra()
        >>> stanoise.average_sta_spectra()
        >>> tfnoise = TFNoise(stanoise)
        >>> tfnoise.transfer_func()
        >>> tfnoise.transfunc.keys()
        dict_keys(['ZP', 'Z1', 'Z2-1', 'ZP-21'])
        """

        # WCC added coher dictionary and added calculations necessary for it
        # Checked ZP-H and found bugs (squashed): haven't checked the others yet!
        transfunc = self.TfDict()
        coher = self.TfDict()

        for key, value in self.tf_list.items():

            if key == 'ZP':
                if value:
                    tf_ZP = {'TF_ZP': self.cZP/self.cPP}
                    coh_ZP = {'TF_ZP': utils.coherence(self.cZP,
                                                       self.cPP, self.cZZ)}
                    transfunc.add(key, tf_ZP)
                    coher.add(key, coh_ZP)

            elif key == 'Z1':
                if value:
                    tf_Z1 = {'TF_Z1': np.conj(self.c1Z)/self.c11}
                    coh_Z1 = {'TF_Z1': utils.coherence(np.conj(self.c1Z),
                                                       self.c11, self.cZZ)}
                    transfunc.add(key, tf_Z1)
                    coher.add(key, coh_Z1)

            elif key == 'Z2-1':
                if value:
                    lc1c2 = np.conj(self.c12)/self.c11
                    coh_12 = utils.coherence(self.c12, self.c11, self.c22)
                    coh_1Z = utils.coherence(self.c1Z, self.c11, self.cZZ)

                    gc2c2_c1 = self.c22*(1. - coh_12)
                    gcZcZ_c1 = self.cZZ*(1. - coh_1Z)
                    gc2cZ_c1 = np.conj(self.c2Z) - np.conj(lc1c2*self.c1Z)

                    lc2cZ_c1 = gc2cZ_c1/gc2c2_c1

                    tf_Z2_1 = {'TF_21': lc1c2, 'TF_Z2-1': lc2cZ_c1}
                    coh_Z2_1 = {'TF_21': np.conj(coh_12),
                                'TF_Z2-1': utils.coherence(gc2cZ_c1, gc2c2_c1,
                                                           gcZcZ_c1)}
                    transfunc.add(key, tf_Z2_1)
                    coher.add(key, coh_Z2_1)

            elif key == 'ZP-21':
                if value:
                    lc1cZ = np.conj(self.c1Z)/self.c11
                    lc1c2 = np.conj(self.c12)/self.c11
                    lc1cP = np.conj(self.c1P)/self.c11

                    coh_Z1 = utils.coherence(self.c1Z, self.c11, self.cZZ)
                    coh_12 = utils.coherence(self.c12, self.c11, self.c22)
                    coh_1P = utils.coherence(self.c1P, self.c11, self.cPP)

                    gc2c2_c1 = self.c22*(1. - coh_12)
                    gcPcP_c1 = self.cPP*(1. - coh_1P)
                    gcZcZ_c1 = self.cZZ*(1. - coh_Z1)

                    gc2cZ_c1 = np.conj(self.c2Z) - np.conj(lc1c2*self.c1Z)
                    gcPcZ_c1 = self.cZP - np.conj(lc1cP*self.c1Z)

                    gc2cP_c1 = np.conj(self.c2P) - np.conj(lc1c2*self.c1P)

                    lc2cP_c1 = gc2cP_c1/gc2c2_c1
                    lc2cZ_c1 = gc2cZ_c1/gc2c2_c1

                    coh_c2cP_c1 = utils.coherence(gc2cP_c1, gc2c2_c1,
                                                  gcPcP_c1)
                    coh_c2cZ_c1 = utils.coherence(gc2cZ_c1, gc2c2_c1,
                                                  gcZcZ_c1)

                    gcPcP_c1c2 = gcPcP_c1*(1. - coh_c2cP_c1)
                    gcZcZ_c1c2 = gcZcZ_c1*(1. - coh_c2cZ_c1)

                    gcPcZ_c1c2 = gcPcZ_c1 - np.conj(lc2cP_c1)*gc2cZ_c1

                    lcPcZ_c2c1 = gcPcZ_c1c2/gcPcP_c1c2

                    coh_cPcZ_c2c1 = utils.coherence(gcPcZ_c1c2, gcZcZ_c1c2,
                                                    gcPcP_c1c2)

                    tf_ZP_21 = {'TF_Z1': lc1cZ, 'TF_21': lc1c2,
                                'TF_P1': lc1cP, 'TF_P2-1': lc2cP_c1,
                                'TF_Z2-1': lc2cZ_c1,
                                'TF_ZP-21': lcPcZ_c2c1}
                    coh_ZP_21 = {'TF_Z1': coh_Z1, 'TF_21': coh_12,
                                 'TF_P1': coh_1P, 'TF_P2-1': coh_c2cP_c1,
                                 'TF_Z2-1': coh_c2cZ_c1,
                                 'TF_ZP-21': coh_cPcZ_c2c1}
                    transfunc.add(key, tf_ZP_21)
                    coher.add(key, coh_ZP_21)

            elif key == 'ZH':
                if value:
                    tf_ZH = {'TF_ZH': np.conj(self.cHZ)/self.cHH}
                    coh_ZH = {'TF_ZH': utils.coherence(np.conj(self.cHZ),
                                                       self.cHH, self.cZZ)}

                    transfunc.add('ZH', tf_ZH)
                    coher.add('ZH', coh_ZH)

            elif key == 'ZP-H':
                if value:
                    lcHcP = np.conj(self.cHP)/self.cHH
                    coh_HP = utils.coherence(self.cHP, self.cHH, self.cPP)
                    gcPcP_cH = self.cPP*(1. - coh_HP)
                    gcPcZ_cH = self.cZP - np.conj(lcHcP*self.cHZ)
                    lcPcZ_cH = gcPcZ_cH/gcPcP_cH
                    tf_ZP_H = {'TF_PH': lcHcP, 'TF_ZP-H': lcPcZ_cH}
                    transfunc.add(key, tf_ZP_H)
                    
                    # coh_HZ = utils.coherence(self.cHZ, self.cHH, self.cZZ)
                    # gcZcZ_cH = self.cZZ*(1. - coh_HZ)
                    coh_cPcZ_cH = utils.coherence(gcPcZ_cH, gcPcP_cH, gcZcZ_cH)
                    coh_ZP_H = {'TF_PH': np.conj(coh_HP), 'TF_ZP-H': coh_cPcZ_cH}
                    coher.add(key, coh_ZP_H)

            elif key == 'ZH-P':
                if value:
                    lcPcH = self.cHP/self.cPP
                    coh_PH = utils.coherence(np.conj(self.cHP), self.cPP, self.cHH)
                    coh_ZH = utils.coherence(np.conj(self.cHZ), self.cZZ, self.cHH)
                    gcHcH_cP = self.cHH*(1. - np.conj(coh_HP))
                    gcZcZ_cP = self.cZZ*(1. - coh_ZP)
                    gcHcZ_cP = np.conj(self.cHZ) - np.conj(lcPcH*np.conj(self.cZP))
                    lcHcZ_cP = gcHcZ_cP/gcHcH_cP
                    coh_cHcZ_cP = utils.coherence(gcHcZ_cP, gcHcH_cP, gcZcZ_cP)
                    tf_ZH_P = {'TF_HP': lcPcH, 'TF_ZH-P': lcHcZ_cP}
                    coh_ZH_P = {'TF_HP': coh_HP,
                                'TF_ZH-P': coh_cHcZ_cP}
                    transfunc.add(key, tf_ZH_P)
                    coher.add(key, coh_ZH_P)

            else:
                raise(Exception('Incorrect key'))

            self.transfunc = transfunc
            self.coher = coher

    def plot(self, key, coher_too=True):
        """Plot all transfer functions for one key
        
        Arguments:
            key (str): TFNoise transfer function key
            coher_too (bool): draw coherency on the same plot
        Returns:
            (numpy.ndarray): array of axis pairs (amplitude, phase)
        """
        if key not in self.transfunc:
            ValueError('key "{key}" not in self.transfunc')
        n_subkeys = len(self.transfunc[key])
        if n_subkeys == 1:
            rows, cols = 1, 1
        elif n_subkeys == 2:
            rows, cols = 1, 2
        elif n_subkeys <= 4:
            rows, cols = 2, 2
        elif n_subkeys <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        ax_array = np.ndarray((rows, cols), dtype=tuple)
        fig, axs = plt.subplots(rows, cols, sharex=True)
        for subkey, i in zip(self.transfunc[key], range(n_subkeys)):
            i_row = int(i/cols)
            i_col = i - cols*i_row
            axa, axp = self.plot_one(key, subkey, fig,
                                     (rows, cols),
                                     (i_row, i_col),
                                     ylabel=i_col == 0,
                                     xlabel=i_row == rows-1,
                                     coher_too=coher_too)
            ax_array[i_row, i_col] = (axa, axp)
        return ax_array

    def plot_one(self, key, subkey, fig=None, fig_grid=(1, 1),
                 plot_spot=(0, 0), xlabel=True, ylabel=True,
                 coher_too=True):
        """Plot one transfer function
        
         Arguments:
            key (str): TFNoise transfer function key
            subkey (str): TFNoise transfer function subkey (possible values
                          depend on key)
            fig (:class: ~matplotlib.figure.Figure): figure to plot on, if 
                None this method will plot on the current figure or create
                a new figure.
            fig_grid (tuple): this plot sits in a grid of this many
                              (rows, columns)
            subplot_spot (tuple): put this plot at this (row,column) of
                                  the figure grid
            xlabel (bool): put an xlabel on this subplot
            ylabel (bool): put a y label on this subplot
            coher_too (bool): draw coherency on the same plot

         Returns:
            tuple:
                transfer function amplitude plot
                transfer function phase plot
            """
        tf = self.transfunc[key][subkey].copy()
        if fig is None:
            fig = plt.gcf()
        # Plot amplitude
        # print(f'{subkey=}, {plot_spot=}')
        fig.suptitle(key)
        ax_a = plt.subplot2grid((3*fig_grid[0], 1*fig_grid[1]),
                              (3*plot_spot[0]+0, plot_spot[1]+0),
                              rowspan=2)
        if coher_too:
            ax2 = ax_a.twinx()
            ax2.semilogx(self.f, np.abs(self.coher[key][subkey]),
                         color='red', linewidth=0.5, alpha=0.8)
            ax2.axhline(np.sqrt(2/self.n_wins),color='red', linewidth=0.5, alpha=0.8, ls='--')
            ax2.set_ylim(0, 1)
            if plot_spot[1]==fig_grid[1]-1:  # Rightmost column
                ax2.set_ylabel('Coher', color='red')
            else:
                ax2.set_yticklabels([])
        tf[tf==0] = None
        # print(f'{self.f[0:2]=},{self.f[-2:-1]=}')
        ax_a.loglog(self.f, np.abs(tf), label=subkey)
        ax_a.set_xlim(self.f[1],self.f[-1])

        legend_1 = ax_a.legend()
        if coher_too:
            legend_1.remove()
            ax2.add_artist(legend_1)
        if ylabel:
            ax_a.set_ylabel('TF')
        else:
            ax_a.set_yticklabels([])
        ax_a.set_xticklabels([])
        # Plot phase
        ax_p = plt.subplot2grid((3*fig_grid[0], 1*fig_grid[1]),
                              (3*plot_spot[0]+2, plot_spot[1]+0))
        ax_p.semilogx(self.f, np.degrees(np.angle(tf)), marker='.', linestyle='')
        ax_p.set_ylim(-180, 180)
        ax_p.set_xlim(self.f[1],self.f[-1])
        ax_p.set_yticks((-180, 0, 180))
        if ylabel:
            ax_p.set_ylabel('Phase')
        else:
            ax_p.set_yticklabels([])
        if xlabel:
            ax_p.set_xlabel('Frequency (Hz)')
        return ax_a, ax_p

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

        >>> from obstools.atacr import DayNoise, StaNoise, TFNoise
        >>> daynoise = DayNoise('demo')
        Uploading demo data - March 04, 2012, station 7D.M08A
        >>> daynoise.QC_daily_spectra()
        >>> daynoise.average_daily_spectra()
        >>> tfnoise_day = TFNoise(daynoise)
        >>> tfnoise_day.transfer_func()
        >>> stanoise = StaNoise('demo')
        Uploading demo data - March 01 to 04, 2012, station 7D.M08A
        >>> stanoise.QC_sta_spectra()
        >>> stanoise.average_sta_spectra()
        >>> tfnoise_sta = TFNoise(stanoise)
        >>> tfnoise_sta.transfer_func()

        Save object

        >>> tfnoise_day.save('tf_daynoise_demo.pkl')
        >>> tfnoise_sta.save('tf_stanoise_demo.pkl')

        Check that everything has been saved

        >>> import glob
        >>> glob.glob("./tf_daynoise_demo.pkl")
        ['./tf_daynoise_demo.pkl']
        >>> glob.glob("./tf_stanoise_demo.pkl")
        ['./tf_stanoise_demo.pkl']

        """

        if not self.transfunc:
            print("Warning: saving before having calculated the transfer "
                  "functions")

        # Remove traces to save disk space
        del self.c11
        del self.c22
        del self.cZZ
        del self.cPP
        del self.cHH
        del self.cHZ
        del self.cHP
        del self.c12
        del self.c1Z
        del self.c1P
        del self.c2Z
        del self.c2P
        del self.cZP
        file = open(filename, 'wb')
        pickle.dump(self, file)
        file.close()
