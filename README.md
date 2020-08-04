# Replication of Recanatesi et al. (2015) "Neural Network Model of Memory Retrieval"

Replication authors: [Carlos de la Torre-Ortiz](https://github.com/c-torre) and [Aurélien Nioche](https://github.com/AurelienNioche/).

Original article: S. Recanatesi, M. Katkov, S. Romani, M. Tsodyks, Neural Network Model of Memory Retrieval, Frontiers in Computational Neuroscience. 9 (2015) 149. [doi:10.3389/fncom.2015.00149](https://doi.org/10.3389/fncom.2015.00149).

Recanatesi *et al.* present a model of memory retrieval based on a Hopfield model for associative learning, with network dynamics that reflect associations between items due to semantic similarities.


## Dependencies

Python 3 with packages in `requirements`.

```pip install -r requirements```
 

## Usage

Clone this repository:

```git clone https://github.com/c-torre/replication-recanatesi-2015.git```

Change the parameters of `main.py` to your needs.
Examples for the parameter sweeps are included in the file.
You may want to lower `T_TOT` for the first figures, and then set it back to normal for cluster simulations.
You should also change the where to save the simulations.
Paths in `paths.py` are automatically recognized for plotting.

Job files for the `slurm` workload manager are included.
`SBATCH` parameters give an idea of the computational cost, and may be tweaked for different cluster requirements.
`run.sh` adds some convenience and requires an `out` directory at project root.

To start simulating networks in parallel:

```./run.sh simulate.job```

Some machines may be able to run `plot.py` with local hardware.
Otherwise run:

```./run.sh plot.job```

Default `main.py`/`simulate.job` outputs are directed to the `simulations` directory tree.
Plotting `plot.py`/`plot.job` outputs are directed to `figures` directory.
All seeds and parameters are saved in a `./simulations/XXXX/parameters.csv` as you decide to produce plots with those simulations for repoducibility.

## Structure

```
├── parameters                               # different model utilities
│   ├── ranges.csv                           # plotting functions
│   └── simulation.csv                       # sine wave function
├── simulations                              # different model utilities
│   └── *param*                              # will contain simulation results
│        └── parameters.csv                  # simulated parameters used for plotting
├── .gitignore                               # clean mess
├── file_loader.py                           # checks files before plotting
├── LICENSE                                  # GPLv3   
├── main.py                                  # runs simulations
├── metadata.tex                             # metadata file for ReScience
├── paths.py                                 # manages project paths
├── plot.job                                 # plotting at cluster
├── plot.py                                  # plotting functions
├── re-neural-network-model-of-...pdf        # replication paper
├── README.md                                # readme; you are here
├── recall_performance.py                    # utils for recall analysis plotting
├── requirements                             # pip install -r requirements
├── run.sh                                   # convenient job runner
└── simulate.job                             # simulating in parallel at cluster
```

## License

[The GNU General Public License version 3](https://www.gnu.org/licenses/#GPL)


## Software Environment

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
