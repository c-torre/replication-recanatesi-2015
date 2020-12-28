[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4396275.svg)](https://doi.org/10.5281/zenodo.4396275)

Replication of Recanatesi et al. (2015) "Neural Network Model of Memory Retrieval"
==================================================================================

Replication authors: [Carlos de la Torre-Ortiz](https://github.com/c-torre) and [Aurélien Nioche](https://github.com/AurelienNioche/).

Original article: S. Recanatesi, M. Katkov, S. Romani, M. Tsodyks, Neural Network Model of Memory Retrieval, Frontiers in Computational Neuroscience. 9 (2015) 149. [doi:10.3389/fncom.2015.00149](https://doi.org/10.3389/fncom.2015.00149).

Recanatesi *et al.* present a model of memory retrieval based on a Hopfield model for associative learning, with network dynamics that reflect associations between items due to semantic similarities.


Dependencies
------------

Python 3 with packages in `requirements.txt`.

```
$ pip install -r requirements
```

Usage
-----

Clone this repository:

```
$ git clone https://github.com/c-torre/replication-recanatesi-2015.git
```

Change the parameters of `simulation.py` to your needs.
You should also change the where to save the simulations.
Some related hints and examples included in `simulation.py` itself.

The project requires a lot of memory and may not run in some regular machines.
Job files for the `slurm` workload manager in a cluster are included.
`SBATCH` parameters inside `.job` files give an idea of the resource requirements, and may be tweaked for different cluster specifications.
`run.sh` is added for convenience, and redirects `slurm` job logs to `./debug/`.

To start simulating networks in parallel:

```
$ sh run.sh simulate.job
```

Some machines may be able to run `plot.py` with local hardware.
Otherwise run:

```
$ sh run.sh plot.job
```

Default `simulation.py`/`simulation.job` outputs are directed to the `simulations` directory tree.
Plotting `plot.py`/`plot.job` outputs are directed to `figures` directory.

Notes:

* You may want to lower `T_TOT` for the first figures, and then set it back to normal for cluster simulations.
* Running `simulation.py` with seed `33` will plot the first figures of the paper (default if run manually rather than with a cluster).
* All seeds and parameters are saved in a `./simulations/XXXX/parameters.csv` as you decide to produce plots with those simulations for repoducibility.

Structure
---------

```
├── figures                           # figures are saved here
├── logs                              # cluster slurm debug information
├── parameters                        # all values for parameters
│   ├── ranges.csv                    # parameter sweeps for figures
│   └── simulation.csv                # parameters for main simulations
│
├── settings                          # developer settings
│   └── paths.py                      # project file paths
│
├── simulations                       # will contain simulation results
│   └── *param*                       # parameters; recalls and patters binaries
│        └── parameters.csv           # simulated parameters used for plotting
│
├── utils                             # misc functions
│   ├── file_loading.py               # loads and checks files before plotting
│   ├── plots.py                      # helper functions for plotting
│   ├── recall_performance.py         # utils for recall analysis plotting
│   └── simulation.csv                # parameters for main simulations
│
├── .gitignore                        # files ignored by git
├── LICENSE                           # GPLv3
├── plot.job                          # plotting at cluster
├── plot.py                           # plotting functions
├── re-neural-network-model...pdf     # replication paper
├── README.md                         # readme; you are here
├── requirements.txt                  # pip install -r requirements
├── run.sh                            # convenient job runner
├── simulation.job                    # simulate model in parallel at cluster
└── simulation.py                     # runs simulations and some plots
```

License
-------

[The GNU General Public License version 3](https://www.gnu.org/licenses/#GPL)

Software Environment
--------------------

Local:

```
Parabola GNU/Linux-libre-5.7.10 x86_64
Python: 3.8.5
```

Cluster:

```
CentOS GNU/Linux 7.8.2003
Python 3.7.7
```

The code is mostly compliant with `pylint`, and always formatted with `black`.
