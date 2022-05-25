# Probabilistic models for multiple-choice (MC) test analysis

## Introduction

A hierarachical series of probablistic models of psychometric states (e.g., knowledge, guessing, blunder) in MC testing.

###  Original citation

Dubins et al CPTL 2016; 8, 598

## Description of python scripts

Files with names pymc3 *.py are scripts for analyzing data (see schematics below):

**pymc3 dubins0_analyze.py**: Knowledge resolution based solely on test scores  
**pymc3 dubins1_analyze.py**: Explicit knowledge based solely on test scores  
**pymc3 categorical analyze.py**: Knowledge resolution with paired confidence ratings

![Schematic of the models](./images/models.png?raw=true "Schematic of the models")

### Notes in usage
a) Currently designed to be run locally. Sample data is provided inline.  
b) Requires PyMC3 3.11.x or newer and its dependencies e.g., theano-pymc.  
c) SciPy is required dependency for MLE fits.
  
## MCQ Simulator

MCQ Simulator is an Excel spreadsheet for computing statistically controlled pass marks based on user inputs (which may come from data analysis from the scripts).

Instructions for use are provided in the first sheet of the spreadsheet.
