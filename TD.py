from blackjack import BlackJack
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
import numpy as np

# Actions
STAND = 0
HIT = 1

# Terminal State
END_STATE = [0, 0]
LOSING_SCORE = -1
WINNING_SCORE = 1
TIE_SCORE = 0


class TD:
    def __init__(self, episodes=1000000, gamma=0.99, learning_rate=None):
        self.blackjack = BlackJack()
        self.value_function = np.zeros((23, 12))
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.episodes = episodes
        self.states_counter = np.zeros((23, 12, 2))

    def step(self, state):
        # Act according to player policy
        # If player score bigger than 17 stand else hit
        if self.blackjack.getPlayerScore() >= 18:
            self.blackjack.standPlayer()
            reward = self.blackjack.score()
            return STAND, reward, END_STATE, True
        else:
            next_state = self.blackjack.hitPlayer()

            # Exceed limit? return lose
            if self.blackjack.getPlayerScore() > 21:
                return HIT, LOSING_SCORE, END_STATE, True

            return HIT, TIE_SCORE, next_state, False

    def update(self, state, action, reward, next_state, finished):
        # Increment by one number of appearance of state
        self.states_counter[state[0], state[1], action] += 1

        # Calculate error
        if not finished:
            error = reward + self.gamma * self.value_function[next_state[0], next_state[1]] - self.value_function[
                state[0], state[1]]
        else:
            error = reward - self.value_function[state[0], state[1]]

        # Calculate alpha
        alpha = 1 / self.states_counter[
            state[0], state[1], action] if self.learning_rate is None else self.learning_rate

        # Update value function according to TD(0)
        self.value_function[state[0], state[1]] += alpha * error

    def run(self):
        wins_number = 0
        for i in range(self.episodes):
            finished = False
            state = self.blackjack.start_game()

            while not finished:
                # Play a action
                action, reward, next_state, finished = self.step(state)

                # Update value function
                self.update(state, action, reward, next_state, finished)
                state = next_state

            # Count number of wins
            if reward == 1:
                wins_number += 1

        return wins_number, self.value_function


def plot(value_function):
    # Build X and Y axis
    min_x = 4
    max_x = 21
    min_y = 2
    max_y = 11
    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z = np.apply_along_axis(lambda _: value_function[(_[0], _[1], True)], 2, np.dstack([X, Y]))
    Z = Z.reshape(X.shape)

    # Build figure
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=cm.coolwarm, vmin=-1.0, vmax=1.0)

    # Set labels and titles
    ax.set_xlabel('Player sum')
    ax.set_ylabel('Dealer showing')
    ax.set_zlabel('Value')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Value function")

    # Set viewing angle
    ax.view_init(ax.elev, 120)

    # Add colorbar
    fig.colorbar(surf)

    # Display the plot
    plt.show()


if __name__ == '__main__':
    gamma = 0.99
    episodes = 5000000

    # Run TD algorithm
    TD = TD(episodes, gamma)
    wins_number, value_function = TD.run()

    # Calculate probability
    prob = wins_number / episodes
    print(f"Win probability is {prob:.4f} for {episodes}")

    # Plot graph
    plot(value_function)

    # Print value function
    # np.set_printoptions(precision=3)
    # for line in value_function:
    #     print(np.around(line,3))
