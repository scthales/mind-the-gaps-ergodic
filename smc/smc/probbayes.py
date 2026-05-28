""" Defines a class for capturing Probability Distributions """

import numpy as np
from scipy.fftpack import dct
from itertools import product
from math import cos, pi
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from smc.probdist import ProbDist

class AgentEnvironment(ProbDist):
    def __init__(self, initial_position=(0, 0)):

        super(AgentEnvironment, self).__init__()

        if self.Nx != self.Ny:
            raise ValueError("Nx and Ny must be equal")

        self.size = self.Nx # since they are equal
        self.grid = np.zeros((self.size, self.size)) # Initialize grid environment
        self.particle_position = initial_position    # Initial position of the particle

        # initialize transition probabilities with uniform distribution
        self.transition_probs = {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}
        # initialize conditional likelihoods
        self.conditional_likelihoods = {}
        # initialize rationality belief for each rationality value
        self.rationality_belief = {0.1: 0.2, 0.3: 0.2, 0.5: 0.2, 0.7: 0.2, 0.9: 0.2}
        self.rationality_belief_post = {}
        print("AgentEnvironment initialized")


    def get_rationality_value(self):
        """
        Get the rationality value based on the posterior probabilities.
        """
        rationality = max(self.rationality_belief, key=self.rationality_belief.get)
        return rationality

    def bayesian_update(self, action):
        """
        Update the posterior probabilities based on an action and likelihoods.
        
        Args:
        action (str): action/outcome observed.
        """
        if action not in self.transition_probs.keys():
            raise ValueError("Invalid action index.")

        # compute the denominator/ marginal likelihood
        marginal_likelihood = sum(self.conditional_likelihoods[rationality][action] * self.rationality_belief[rationality] for rationality in self.conditional_likelihoods)

        for rationality in self.conditional_likelihoods:
            self.rationality_belief_post[rationality] = (self.conditional_likelihoods[rationality][action] * self.rationality_belief[rationality]) / marginal_likelihood

        self.rationality_belief = self.rationality_belief_post.copy()


    def get_conditional_likelihoods(self, actions, action_qvalues, rationality_values):
        """
        Compute conditional likelihoods for each value of the variable given an action.
        eq. 2
        conditional likelihood is the probability of the action given the rationality value
        is a dictionary if rationality value that gives the probability of each action
        """
        prob_actions = {}
        # each rationality value has a list of probabilities for each action
        for rationality in rationality_values:
            # convert dict to list of values
            # compute exponentials and sum
            qvalues = np.array(list(action_qvalues.values()))
            sum_qvalues = np.sum(np.exp(rationality * qvalues))

            # compute conditional likelihoods for each value of the variable
            for action in actions:
                prob_actions[action] = np.exp(rationality * action_qvalues[action]) / sum_qvalues

            self.conditional_likelihoods[rationality] = prob_actions.copy()

    def get_action_probabilities(self, curr_rationality):
        """
        copy the transition probabilities given rationality level
        """
        self.transition_probs = self.conditional_likelihoods[curr_rationality].copy()
        

    def move_particle(self, action):
        # Implement particle movement based on transition probabilities
        if action == 'up':
            self.particle_position = (self.particle_position[0], min(self.size - 1, self.particle_position[1] + 1))
        elif action == 'down':
            self.particle_position = (self.particle_position[0], max(0, self.particle_position[1] - 1))
        elif action == 'left':
            self.particle_position = (max(0, self.particle_position[0] - 1), self.particle_position[1])
        elif action == 'right':
            self.particle_position = (min(self.size - 1, self.particle_position[0] + 1), self.particle_position[1])


    def get_probabilities(self, current_position):
        # Get prediction probabilities for next steps based on current position
        prediction_probs = {
            'up': self.transition_probs['up']       if current_position[1] < self.size - 1 else 0,
            'down': self.transition_probs['down']   if current_position[1] > 0 else 0,
            'left': self.transition_probs['left']   if current_position[0] > 0 else 0,
            'right': self.transition_probs['right'] if current_position[0] < self.size - 1 else 0
        }
        return prediction_probs


    def get_prediction_probabilities(self):
        '''
        Get prediction probabilities for all next steps with probl limited
        by limit probability from current position
        '''

        # initialize grid probabilities with zeros
        grid_data = np.zeros((self.size, self.size))
        # the position varies between 0 and 1, and is normilized the self.size
        grid_data[int(self.particle_position[0]), int(self.particle_position[1])] = 1.0

        # start computing probabilities from the current position
        current_position = self.particle_position 
        visited_positions = []
        visite_neighbors = [[current_position, 1.0]]
        limit_prob = 0.01
        # then move around the current position to compute the probabilities
        # the steps should be considered here
        #breakpoint()

        while len(visite_neighbors) > 0:
            current_position, current_prob = visite_neighbors.pop(0)
            visited_positions.append(current_position)

            prediction_probs = self.get_probabilities(current_position)

            for action in prediction_probs:
                # this is moving x/x and y are inverted (y, x)
                if action == 'up':
                    next_position = (current_position[0], min(self.size - 1, current_position[1] + 1))
                    next_prob = current_prob * prediction_probs[action]
                elif action == 'down':
                    next_position = (current_position[0], max(0, current_position[1] - 1))
                    next_prob = current_prob * prediction_probs[action]
                elif action == 'left':
                    next_position = (max(0, current_position[0] - 1), current_position[1])
                    next_prob = current_prob * prediction_probs[action]
                elif action == 'right':
                    next_position = (min(self.size - 1, current_position[0] + 1), current_position[1])
                    next_prob = current_prob * prediction_probs[action]


                # if the next position is not visited, add it to the list of positions to be visited
                # it actually, should visit some points more than once
                total_prob = min(grid_data[next_position[0], next_position[1]] + next_prob, 1.0)

                if total_prob > limit_prob:
                    grid_data[next_position[0], next_position[1]] = total_prob

                if next_prob > limit_prob:
                    visite_neighbors.append([next_position, next_prob])

        return grid_data


    def plot_environment(self):
        # Plot grid environment with color-coded probabilities around the particle
        # >>>>> This function should give the pred. prob for the whole environment around the particle
        grid_data = self.get_prediction_probabilities()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(grid_data, cmap='Blues', origin='upper', vmin=0, vmax=1)
        plt.colorbar(label='Prediction Probability')
        plt.scatter(self.particle_position[1], self.particle_position[0], color='red', s=50, label='Target Position')
        plt.legend()
        plt.title('Grid Environment with Prediction Probabilities')

        plt.text(12.8, 2, 'Beta--Confidence on the model = 0.1', color='black', bbox=dict(facecolor='white', boxstyle='round', alpha=0.1))

        plt.xlim(-0.5, self.size - 0.5)
        plt.ylim(self.size - 0.5, -0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('my_plot2.png')
        plt.show()
