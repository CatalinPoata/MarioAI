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


def hash_state(old_state: numpy.ndarray):
    return binascii.crc32(numpy.ndarray.tobytes(numpy.ndarray.flatten(old_state))) & 0xFFFFFF


movement_type = SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, movement_type)
env = GrayScaleObservation(env, keep_dim=True)

state = env.reset()

action_size = len(movement_type)
state_size = 256 * 256 * 256
qtable = np.fromfile("last", dtype=float).reshape(state_size, action_size)

print(qtable.shape)
print(np.count_nonzero(qtable))

max_steps = 60 * 400

for step in range(max_steps):
    hashed_state = hash_state(state)
    # action_weights = [i+1 for i in qtable[hashed_state]]
    if np.all(qtable[hashed_state] < 0):
        action_weights = [i + qtable[np.argmin(qtable[hashed_state])] + 1 for i in qtable[hashed_state]]
        action = random.choices(range(len(movement_type)), weights=action_weights, k=1)[0]
    else:
        action = np.argmax(qtable[hashed_state])

    print(action)
    new_state, reward, done, info = env.step(action)
    env.render()
    state = new_state
    sleep(100 / 1000000.0)

    if done:
        break
