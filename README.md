# tsc-2019

This repository is for an agent we presented at [The 2019 Science of Consciousness conference](https://www.tsc2019-interlaken.ch). 
Also see our [slides](docs/TSC-2019_slides.pdf) from the conference for more information and context.

The DQN agent, model, and training code was borrowed heavily from the repository [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn).

## Setup

1. Install pipenv: `pip3 install pipenv`
1. Set up the environment with the required packages: `pipenv install
1. Install and enable the jupyter notebook extensions (optional):
```
pipenv run jupyter contrib nbextension install --user
pipenv run jupyter nbextension enable codefolding/main
pipenv run jupyter nbextension enable toc2/main
pipenv run jupyter nbextension enable collapsible_headings/main
pipenv run jupyter nbextension enable varInspector/main
pipenv run jupyter nbextension enable spellchecker/main
``` 


## Usage

## Running the notebook

1. Launch a jupyter notebook server: `pipenv run jupyter notebook`
1. Open and run the notebook located in `notebooks/TSC-2019.ipynb`


## Training the model

To train the model run: `pipenv run training.py`

