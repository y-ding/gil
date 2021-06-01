# Generalizable and Interpretable Learning for Configuration Extrapolation

This is the Python implementation of GIL and GIL+ described in:
 
Yi Ding, Ahsan Pervaiz, Michael Carbin, and Henry Hoffmann. 
[Generalizable and Interpretable Learning for Configuration Extrapolation](https://y-ding.github.io/xxx). In the 29th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 2021).

## License

This code is licensed under the MIT license, as found in the LICENSE file.

## Requirements
* Python (>=3.6)
* scikit-learn

## Data
The `data` folder includes data for 10 HiBench workloads with 2000 configurations each on 3 hardware (skylake, haswell, storage):
* `perf`: application configurations, performance (throughput).
* `llsm`: low-level system metrics.

## Experiments

### low2high

#### Extrapolation example:

`python extrapolate_low2high.py --workload als --outcome throughput --n_start 200 --n_step 20 --n_iter 20`

#### Interpretation example:

`python interpret_low2high.py --workload als --outcome Throughput --n_start 200 --n_step 20 --n_iter 20`

### mid2high

#### Extrapolation example:

`python extrapolate_mid2high.py --target has --workload als --outcome Throughput --n_start 300 --n_step 20 --n_iter 10`

#### Interpretation example:

`python interpret_mid2high.py --target has --workload als --outcome Throughput --n_start 200 --n_step 20 --n_iter 10`







