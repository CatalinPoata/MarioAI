import os
import random
from time import sleep

import numpy
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation
import binascii

movement_type = RIGHT_ONLY


def normalize_reward(old_reward):
    if old_reward <= 0:
        return -1
    return (((old_reward - -15) * (1 - 0)) / (15 - -15)) + 0


def hash_state(old_state: numpy.ndarray):
    return binascii.crc32(numpy.ndarray.tobytes(numpy.ndarray.flatten(old_state))) & 0xFFFFFF


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, movement_type)
env = GrayScaleObservation(env, keep_dim=True)

state = env.reset()

lastFile = open("last", "w")
lastWinner = open("lastWinner", "w")

print(state.shape)
print(type(state))

action_size = len(movement_type)
state_size = 256 * 256 * 256  # 240 x 256 resolution + grayscale = 256 possible color variations
# -> 256 * 240 * 256 possible pixel combinations on screen
# rounded it to 256 * 256 * 256 for 24 bit max, lesser chances for a hash collision

qtable = np.zeros((state_size, action_size))
print(qtable.shape)
print("Non-zeroes: ", numpy.count_nonzero(qtable))
print(qtable)

total_episodes = 10000000
learning_rate = 0.7
max_steps = 60 * 400  # 400s @ 60fps
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.0005

rewards = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        tradeoff = random.uniform(0, 1)

        if tradeoff > epsilon:
            action = np.argmax(qtable[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # print(reward)

        env.render()

        reward = normalize_reward(reward)
        hashed_state = hash_state(state)
        hashed_new_state = hash_state(new_state)

        qtable[hashed_state, action] = qtable[hashed_state, action] + learning_rate * (
                reward + gamma * np.max(qtable[hashed_new_state, :]) - qtable[hashed_state, action])
        total_rewards += reward
        # print(total_rewards)
        state = new_state


        if done:
            if info["flag_get"]:
                lastWinner.truncate(0)
                qtable.tofile(lastWinner)
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)
    print(qtable)
    print("Non-zeroes: ", numpy.count_nonzero(qtable))
    if episode % 10 == 0:
        lastFile.truncate(0)
        qtable.tofile(lastFile)
