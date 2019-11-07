# Replication of Recanatesi et al. (2015) "Neural Network Model of Memory Retrieval"

Replication authors: [Carlos de la Torre-Ortiz](https://github.com/c-torre) and [Aurélien Nioche](https://github.com/AurelienNioche/).

Original article: S. Recanatesi, M. Katkov, S. Romani, M. Tsodyks, Neural Network Model of Memory Retrieval, Frontiers in Computational Neuroscience. 9 (2015) 149. [doi:10.3389/fncom.2015.00149](https://doi.org/10.3389/fncom.2015.00149).

Recanatesi *et al.* present a model of memory retrieval based on a Hopfield model for associative learning, with network dynamics that reflect associations between items due to semantic similarities.


## Dependencies

Python 3 with packages:
 * Numpy
 * matplotlib
 * tqdm

```pip3 install --user numpy matplotlib tqdm```
 

## Usage

Clone this repository:

```git clone https://github.com/c-torre/replication-recanatesi-2015.git```

Run *main.py*:

```python3 main.py```

The network will run automatically and generate all plots in the */fig* directory relative to where the script is saved.


## Structure

```
├── tools                                    # different model utilities
│   ├── __init__.py                          
│   ├── plots.py                             # plotting functions
│   └── sine_wave.py                         # sine wave function
├── LICENSE                                  # GPLv3   
├── main.py                                  # <Run this file to start the model>
├── metadata.tex                             # metadata file for ReScience
├── README.md                                # readme; you are here
└── re-neural-network-model-of-...pdf        # replication paper
```


## License

[The GNU General Public License version 3](https://www.gnu.org/licenses/#GPL)


## Software Environment

```
OS: Manjaro GNU/Linux 18.1.2 x86_64
Python: 3.7.4 (default, Oct  4 2019, 06:57:26)
[GCC 9.2.0] on linux
NumPy: 1.17.2
matplotlib: 3.1.1
tqdm 4.28.1
```
Also tested on: Ubuntu GNU/Linux 18 LTS, and MacOS Mojave.
