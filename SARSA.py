from blackjack import BlackJack
from matplotlib import pyplot as plt
from matplotlib import colors

import numpy as np

# Actions
STAND = 0
HIT = 1

# Terminal State
END_STATE = [0, 0]
LOSING_SCORE = -1
WINNING_SCORE = 1
TIE_SCORE = 0


class SARSA:
    def __init__(self, episodes=1000000, gamma=0.99, learning_rate=None, get_action="epsilon", epsilon=None, beta=30):
        self.blackjack = BlackJack()
        self.q_function = np.zeros((23, 12, 2))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.episodes = episodes
        self.beta = beta
        self.actions = [0, 1]
        self.get_action = self.epsilonGreedy if get_action == "epsilon" else self.softmax
        self.states_counter = np.zeros((23, 12, 2))

    def step(self, action):
        # Act according to player action
        if action == HIT:
            next_state = self.blackjack.hitPlayer()

            # Exceed limit? return lose
            if self.blackjack.getPlayerScore() > 21:
                return LOSING_SCORE, END_STATE, True

            return TIE_SCORE, next_state, False
        else:
            self.blackjack.standPlayer()
            reward = self.blackjack.score()
            return reward, END_STATE, True

    def epsilonGreedy(self, state):
        # Calculate epsilon
        epsilon = self.epsilon
        if not self.epsilon:
            count = sum(self.states_counter[state[0], state[1]])
            if count == 0:
                epsilon = 1
            else:
                epsilon = 1 / count

        # Explore or exploit?
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            # Both action are equal? Select randomly
            if self.q_function[state[0], state[1]][0] == self.q_function[state[0], state[1]][1]:
                action = np.random.choice(self.actions)
            else:
                action = np.argmax(self.q_function[state[0], state[1]])

        return action

    def softmax(self, state):
        # Get the Q-values for the given state
        q_values = self.q_function[state[0], state[1]]

        # Apply the softmax function to the Q-values
        exp_q_values = np.exp(self.beta * q_values)
        softmax_probs = exp_q_values / np.sum(exp_q_values)

        # Select the action with the highest softmax probability
        if softmax_probs[0] == softmax_probs[1]:
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(softmax_probs)

        # Return the selected action
        return action

    def update(self, state, action, reward, next_state, next_action, finished):
        # Increment by one number of appearance of state
        self.states_counter[state[0], state[1], action] += 1

        # Calculate error
        if not finished:
            error = reward + self.gamma * self.q_function[next_state[0], next_state[1], next_action] - self.q_function[
                state[0], state[1], action]
        else:
            error = reward - self.q_function[state[0], state[1], action]

        # Calculate alpha
        denominator = (
            (self.states_counter[state[0], state[1], action] + 1)) if self.learning_rate is None else self.learning_rate
        alpha = 1 / denominator

        # Update value function according to TD(0)
        self.q_function[state[0], state[1], action] += alpha * error

    def run(self):
        wins_number = 0
        state_wins_number = np.zeros((23, 12))
        for i in range(self.episodes):
            finished = False

            # First action and state
            state = self.blackjack.start_game()
            action = self.get_action(state)

            # Path
            path = []

            while not finished:
                # Play a action
                reward, next_state, finished = self.step(action)

                # Get next iteration action
                next_action = self.get_action(next_state)

                # Update q function according to SARSA law
                self.update(state, action, reward, next_state, next_action, finished)

                # Add state to path
                path.append(state)

                # Continue to t+1
                action = next_action
                state = next_state

            # Count number of wins
            if reward == 1:
                wins_number += 1

                # Increment states wins
                for state in path:
                    state_wins_number[state[0], state[1]] += 1

        # Calculate probability, policy and number of wins from each state
        optimal_action = np.zeros((23, 12))
        optimal_action_string = np.empty((23, 12), dtype=object)
        probs = np.empty((23, 12), dtype=object)
        for i in range(23):
            for j in range(12):
                optimal_action[i, j] = np.argmax([self.q_function[i, j, action] for action in self.actions])
                optimal_action_string[i][j] = ('stand' if optimal_action[i, j] == 0 else 'hit')
                x = sum(self.states_counter[i, j])
                probs[i][j] = (state_wins_number[i, j] / x)

        return wins_number, optimal_action, optimal_action_string, probs


if __name__ == '__main__':
    gamma = 0.99
    episodes = 50000000

    # Run SARSA
    SARSA = SARSA(episodes, gamma, get_action="greedy")
    wins_number, optimal_action, optimal_action_string, probs = SARSA.run()

    # Print win probability
    prob = wins_number / episodes
    print(f"Win probability is {prob:4f} for {episodes}")

    # Print optimal policy for each state
    for i in range(23):
        for j in range(12):
            if not np.isnan(probs[i][j]):
                print(f"For state ({i}, {j}), win prob {probs[i][j]:4f}, optimal action: {optimal_action_string[i][j]}")