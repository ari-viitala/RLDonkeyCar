# Learning to Drive Small Scale Cars from Scratch

This code base contains the code to learn to drive a Donkey Car using reinforcement learning. This approach uses a combination of a variational autoencoder to learn a small dimensional state representation for image data and soft actor-critic to learn a control policy based on the small dimensional state representation. This implementation is able to learn to follow a track in 15 minutes of driving around on a track which corresponds to about 10000 samples from the environment.


### Core files
`models/ae.py` and `models/ae_sac.py` contain the codes for the autoencoder as well as the sof actor-critic implementations.
`train.py` contains the code for using the agent to drive in the environment as well as training the agent.
`config.py` contains general variables for the environment, the agent and the training process

## Running the code

Install the required libraries using 

```
pip install -r requirements.txt
```

You need to be running the Donkey Car using the remote management features as described in this [project](https://github.com/tawnkramer/learning-to-drive-in-5-minutes).

You can train a car by running command

```
python train.py --env_type DonkeyCar --car_name <your car's name>
```

Set the car on the track and run the command. The program prompts you to input a throttle. For a smooth surface and a full Ni-MHH battery 0.17 is usually fine for a steady constant speed.

Once you enter the throttle the car starts to drive. Once the car drives off the track press Crtl + C to terminate the episode. Set the car back on track. Before the start of the next episode you can adjust the throttle if you wish.

As default the car has five totally random episodes before the start of training. After this actions provided by the agent are used to drive the car and after each episode the agent is updated.
