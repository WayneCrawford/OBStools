WCC modifications:
- Divided classes into individual files
- Added nwins attribute to TFNoise class (needed for n_coher below)
- Added n_coher variable to EventStream.correct_data(): only applies transfer
  function if the coherency is higher than the 95% significance level for n_coher
  neighboring indices
