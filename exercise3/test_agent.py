from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json

from model import Model
from utils import *

history_length = 1

def run_episode(env, agent, history_length, rendering=True, max_timesteps=1000):

    episode_reward = 0
    step = 0

    state = env.reset()
    state_history = np.zeros((1, state.shape[0], state.shape[1], history_length))

    while True:

        # preprocess the state in the same way than in in your preprocessing in train_agent.py
        state = rgb2gray(state)
        state_history[0,:,:,0:history_length-1] = state_history[0,:,:,1:]
        state_history[0,:,:,-1] = state

        if step < 20:
           a = [0.0, 1.0, 0.0]
        else:
           # get the action from your agent! If you use discretized actions you need to transform them to continuous
           # actions again. a needs to have a shape like np.array([0.0, 0.0, 0.0])
           a = agent.session.run(agent.y_pred, feed_dict={agent.x:state_history})[0]

           a_id = np.where(a==max(a))

           #print('action: ', a_id[0])

           if (a_id[0] == 0):
              a = [0.0, 0.0, 0.0]
           elif (a_id[0] == 1):
               a = [-1.0, 0.0, 0.0]
           elif (a_id[0] == 2):
               a = [1.0, 0.0, 0.0]
           elif (a_id[0] == 3):
               a = [0.0, 1.0, 0.0]
           else:
               a = [0.0, 0.0, 0.2]

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        next_state, r, done, info = env.step(a)
        episode_reward += r
        state = next_state
        step += 1

        if rendering:
            env.render()

        if done or step > max_timesteps:
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True
    n_test_episodes = 5
    batch_size = 256
    history_length = 5

    # load agent
    agent = Model(history_length = history_length, batch_size = batch_size, set_to_default = True)
    agent.load("models/agent.ckpt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, history_length = history_length, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()

    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print('... finished')
