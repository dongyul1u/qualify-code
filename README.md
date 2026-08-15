# Qualify Code

This folder has two entry files with different run environments.

## HPC

`qualify_code.py` is the experiment script. Run this on HPC.

`qualify_code.py` runs the experiments, saves the raw results, and generates part of the result-analysis figures.

## Local

`result_analysis.py` is the local replot and result-analysis script. Run this on your local machine.

`result_analysis.py` is used to make fuller use of the experiment results. The figures and tables used for analysis in the experiment mainly come from this script.
