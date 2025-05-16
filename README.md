# filters

The starter code is written in Python and depends on NumPy and Matplotlib.
This README gives a brief overview of each file.

- `localization.py` -- This is your main entry point for running experiments.
- `soccer_field.py` -- This implements the dynamics and observation functions, as well as the noise models for both. Jacobian implementations were added.
- `utils.py` -- This contains assorted plotting functions, as well as a useful
  function for normalizing angles.
- `policies.py` -- This contains a simple policy, which you can safely ignore.
- `ekf.py` -- Kalman filter implementation here!
- `pf.py` -- Particle Filter implementation here!
- `analyze_ekf.py` -- Contains the analysis and plots for Kalman filter.
- `pf_analysis.py` -- Contains the analysis and plots for Partical filter.

## Command-Line Interface

To visualize the robot in the soccer field environment, run
```bash
$ python localization.py --plot none
```
The blue line traces out the robot's position, which is a result of noisy actions.
The green line traces the robot's position assuming that actions weren't noisy.

The filter's estimate of the robot's position will be drawn in red.
This will show also some calculations and plots the analysis.
```bash
$ python analyze_ekf.py --plot ekf 
$ python pf_analysis.py --plot pf 
```

