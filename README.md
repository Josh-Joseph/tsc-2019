# tsc-2019


borrowed heavily from https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn


## old files

train_dqn_with_mental_states.py

train_mental_state_self_report.py

dqn_agent.py

world.py

model.py

## new files

train.py - main script that trains the agent's NN and saves it 





## usage

pipenv install

pipenv run jupyter contrib nbextension install --user

pipenv run jupyter nbextension enable spellchecker/main
pipenv run jupyter nbextension enable codefolding/main
pipenv run jupyter nbextension enable toc2/main
pipenv run jupyter nbextension enable collapsible_headings/main
pipenv run jupyter nbextension enable varInspector/main

pipenv run jupyter notebook



Plan:



* settle on lunar lander training/hyperparams/saved model
* enumerate the mental states 
* run 20 simulations to gather data of [brain state, mental state]
* train mlp