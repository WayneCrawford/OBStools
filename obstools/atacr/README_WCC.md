**I HAVE DECIDED TO PUT THE CODES  DESCRIBED BELOW IN CRAWTOOLS/SPECTRAL SpectralDensity and TransferFunction classes**

I will be writing alternative TfNoise and EventStream scripts to overcome some
problems I have with adding transfer functions and calculating time series:

Transfer Functions:
-------------------

All transfer functions are calculated using prewritten "recipes", instead of
programmatically.  This makes it complicated to calculate new transfer function
combinations and it also makes it easier to make an error.

I am thinking of using XArray to store coherences and spectra (can I use an
on-the-fly calculation for complex conjugate coherences?) and then to calculate transfer
functions as I need them.  The problem is that multi-level transfer functions
have their own coherencies: how to store these?  Maybe transfer functions can be
a subclass of a given coherency-spectra.  ATACR calculates transfer function
at the same time as the coherency, but this doesn't work if you accept that
transfer functions depend on many more assumptions than coherences (which
channel contains the noise, whether to calculate a transfer function if coherence
is insignificant...).

I guess should just expand xarray table to include the new combinations, so in
the beginning we would have:
1, 2, Z, P, L (longitudinal)
and if we removed L from the other channels we would have:
1, 2, Z, P, L, 1-L, 2-L, Z-L, P-L
then if we removed P we would have:
1, 2, Z, P, L, 1-L, 2-L, Z-L, P-L, 1-L-P, 2-L-P, Z-L-P

or would it be easier/more logical to have separate XArrays for each removal?
1, 2, A, P, L
1-L, 2-L, Z-L, P-L
1-L-P, 2-L-P, Z-L-P

I could also have separate spectra/coherences for 1/2 and for L:
1, 2, Z, P
L, Z, P

I could easily add an L^2 to the XArray

A future development could be to allow multivariate coherencies/transfer functions


Time Series:
------------

ATACR calculates cleaned time series in EventStream.correct_data() as the
inverse FFT of the corrected spectra.  This means that the cleaned time series
does not correspond to the input time series?

Current Workflow:
------------

- make a list of DayNoise objects based on daily slices of 1, 2, Z and H data
- make a StaNoise object based on this list (is this ever used aside from plot?)
- make a TFNoise object based on the first DayNoise object
- calculate transfer functions in the TFNoise object
- Make an EventStream object from the data
- Run eventstream.correct_data(TFNoise object)

New classes:
------------------

- `SpecCoher`: calculate and store spectra and coherences
- `TransferFunc`: Calculate transfer function (single, w.r.t. one channel, or for all SpecCohers)
- `CleanCalc`: Calculate cleaning sequence
- `Clean`: apply cleaning sequence to spectra, returns new SpecCoher object
