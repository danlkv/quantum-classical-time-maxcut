## Data collection scripts

- `QtensorVariance` - Exact calculation of QAOA variance for small p=2,3. Uses a single graph seed.
  Generates data `data/qaoa_variance_exact_p23_N256.nc` and
  `data/qaoa_variance_exact_p23.nc` to be used in `Cost variance vs N.ipynb` 

- `QAOA variance estimations` - Approximate QAOA variance calculations using sampling and done for larger p<12
  Generates data `data/qaoa_variance_nsamples1000.nc` to be used in `Cost
  variance vs N.ipynb`.
