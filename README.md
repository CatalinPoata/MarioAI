# MarioAI

This project implements a Q-Learning algorithm using the `gym-super-mario-bros` environment to teach an agent how to play Super Mario Bros.

## Overview

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing certain actions and receiving feedback in the form of rewards or penalties. In this project, we apply Q-Learning, a model-free RL algorithm, to train an agent to navigate and play levels of Super Mario Bros.

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/CatalinPoata/MarioAI.git
   cd MarioAI
   ```

2. **Create and activate a Python virtual environment (optional but recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use 'env\Scripts\activate'
   ```

   Ensure that the `gym-super-mario-bros` environment is correctly installed. For more details, refer to the [gym-super-mario-bros documentation](https://github.com/Kautenja/gym-super-mario-bros).

## Training the Agent

To train the Mario agent using Q-Learning:

```bash
python main.py
```

This script initializes the environment and starts the training process. The agent will learn to play the game by interacting with the environment and updating its Q-values based on the rewards received.

## Running the Trained Agent

After training, you can run the trained agent to see its performance:

```bash
python run.py
```

This script loads the trained Q-values and allows the agent to play the game based on its learned policy.

## Project Structure

- `main.py`: Contains the training loop for the Q-Learning algorithm.
- `run.py`: Loads the trained Q-values and runs the agent in the environment.

## Acknowledgements

This project utilizes the `gym-super-mario-bros` environment, which provides a convenient interface for training RL agents on Super Mario Bros. Special thanks to the contributors of that project.

## References

- [gym-super-mario-bros Documentation](https://github.com/Kautenja/gym-super-mario-bros)
