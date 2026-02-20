# O3-precessing-BBH-search

This repository contains the data products and analysis scripts associated with the paper "Search for Precessing Binary Black Holes in Advanced LIGO's Third Observing Run using Harmonic Decomposition" by Rahul Dhurkunde and Ian Harry (arXiv:2601.04276).

### Introduction

Gravitational-wave (GW) signals from binary black holes (BBHs) with misaligned spins exhibit orbital precession, which provides critical information about their formation history and environment. Standard searches often use aligned-spin template banks, which can lose sensitivity to these complex signals.

This work implements a dedicated search for precessing BBHs in Advanced LIGO's third observing run (O3) data. Using a harmonic decomposition method, the search expresses precessing waveforms as a sum of five harmonics. The repository provides the tools and results for a novel filtering scheme that reduces the required number of templates by 5× while improving sensitivity by up to 28% compared to previous precessing search pipelines.

## Repository Structure

The repository is organized to facilitate the reproduction of results and the use of the precessing search data products:

- search-configs/: Contains the configuration files (e.g., .ini files) used to run the PyCBC-based search pipeline on O3 data. 

- bank/: Includes .ini files to generate the precessing and aligned-spin template banks.

- scripts-for-figures/: Python scripts and Jupyter notebooks for visualizing the results and generating the paper's figures.

## Citation

If you use the data or methods in this repository, please cite the original paper:

```
@article{Dhurkunde:2026precessing,
  title={Search for Precessing Binary Black Holes in Advanced LIGO's Third Observing Run using Harmonic Decomposition},
  author={Dhurkunde, Rahul and Harry, Ian},
  journal={arXiv preprint arXiv:2601.04276},
  year={2026}
}
```
