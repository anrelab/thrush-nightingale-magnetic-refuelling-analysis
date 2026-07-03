# Analysis of Magnetic-Field Effects on Refuelling in Thrush Nightingales

This repository contains Python scripts used for data processing, visualisation, and statistical analysis in a field experiment testing whether a simulated magnetic-field route sequence affects refuelling and body-mass dynamics in juvenile thrush nightingales (*Luscinia luscinia*).

The experiment was conducted in 2024 and 2025 using the same protocol. Juvenile thrush nightingales were assigned either to a control group kept in the natural magnetic field or to an experimental group exposed to a stepwise simulated magnetic-field sequence corresponding to locations along the autumn migratory route, ending in northern Egypt.

The scripts convert the original dataset into tidy long format, calculate body-mass gain relative to Day 0, analyse daily and cumulative food consumption, generate figures, fit pooled mixed-effects models, perform period-specific comparisons of body-mass-gain slopes, calculate interval-based correlations between food consumption and body-mass change, and run bootstrap robustness checks.

The repository is intended to make the analytical workflow transparent and reproducible.
