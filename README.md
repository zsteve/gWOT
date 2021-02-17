# gWOT: Global Waddington-OT

Principled trajectory inference for time-series data with limited samples by optimal transport.

*Important: this README is currently under construction. Check back soon!*

[![PyPI version](https://badge.fury.io/py/gwot.svg)](https://badge.fury.io/py/gwot)

## Introduction

Global Waddington-OT (gWOT) is a trajectory inference method for time-series data based on optimal transport (OT).
Given a time-series of snapshot data, gWOT aims to estimate trajectory information in the form of a _probability distribution_ over possible trajectories taken by cells.

As an example, we illustrate below a ground truth process where cell trajectories are known exactly (green). From this, independent snapshots are sampled at various temporal instants, each with limited sample resolution (red). From these data, gWOT aims to reconstruct trajectories as a law on paths (blue).

![Example sample path reconstruction](aux_files/illustration.png)

The underlying model assumption on which gWOT is based is that the generative process is a drift-diffusion process with branching, in which the evolution of any cell over an infinitesimal time is described by the stochastic differential equation (SDE) 

![Diffusion-drift SDE](aux_files/sde.png).

Cells in this process also divide and die at rates `beta(x, t)` and `delta(x, t)` respectively.

## Installation

To install, use `pip install gwot`.

Alternatively, clone this repository and `cd gWOT && pip install .`

## Example application: bistable landscape with branching
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zsteve/gWOT/blob/main/examples/gWOT_example.ipynb)

## Paper

This code accompanies the paper (TBA on arXiv)
```
Hugo Lavenant, Stephen Zhang, Young-Heon Kim, and Geoffrey Schiebinger.
Towards a mathematical theory of trajectory inference, 2021. 
```
