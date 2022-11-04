# Autonomous-Delivery-Robot

Setup:

Clone github repo
Install Python3.6 (can use virtual environment for this)
Install all required libraries using pip install -r requirements.txt"
Install OpenAI Baselines

git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

Install Python-RVO2 library

Getting started

This repository is organized in different parts:

crowd_sim/ folder contains the simulation environment.

crowd_nav/ folder contains configurations and non-neural network policies

pytorchBaselines/ contains the code for the DSRNN network and ppo algorithm.

a/ folder contains all the functions/classes used for navreptrainenv.py <- the file containing the NavRep training environment


Please try not to modify the a/ folder without first understanding how it affects the NavRep training environment. Because the code is a mashup of two different repositories, there are some odd quirks when running the code-

There are two different files containing the configs for the code
a/crowd_nav/config/test_soadrl_static.config <- used for modifying the NavRep training environment EXCLUSIVELY (e.g. no. pedestrians, robot FOV, etc.)
crowd_nav/configs/config.py <- used for modifying the training/testing stage itself (e.g. no. training steps, whether to visualise training or not, etc.)

Running code

To test our trained model. python -m test --model_dir data/eh12 --test_model 56400.pt --visualize

To Train a policy. python train.py
To Test policies. Please modify the test arguments in the begining of test.py. python test.py
To Plot training curve. python plot.py

# Pedestrian Controlled Algortihms
For pedestrian controlled algorithms testing change branch to pedestrians-only


Pre-requisites
Python 3.6

For example, on Ubuntu 20
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.6 python3.6-dev
sudo apt-get install -y virtualenv python3-virtualenv

We recommend creating a virtualenv:

virtualenv ~/navrepvenv --python=python3.6
source ~/navrepvenv/bin/activate

To run the testing. python3 -m navrep.scripts.test_navrep --backend VAE_LSTM --encoding V_ONLY --render

To change policies, go to test_soadrl_static.config and change the human_policy to Social Forces, ORCA, Linear 
For Randomomised Policies, random_policy_changing should be True
For human number, change human_number variable

For changing radius and velocity of pedestrians go to agent.py and change variables under sample_random_attributes function
