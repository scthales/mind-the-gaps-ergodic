""" Defines abstract class for coverage problems """

import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, pow
from itertools import product

class Coverage(object):
    """ Abstract class for coverage problems """
    def __init__(self, target_distribution):
        """ Constructor for Coverage class """
        self.target_distribution = target_distribution
        self.agents = []

    def add_agent(self, new_agent):
        """ Adds new agent to the coverage problem """
        self.agents.append(new_agent)

    def reflect_agent(self, agent):
        """ Makes sure that agent location is within bounds of target distribution """
        
        dx = self.target_distribution.dx
        dy = self.target_distribution.dy
        xmin = self.target_distribution.xmin
        ymin = self.target_distribution.ymin
        xmax = self.target_distribution.xmax
        ymax = self.target_distribution.ymax
        
        if agent.x < xmin - dx:
            agent.x = xmin + (xmin - agent.x)
        if agent.x > xmax - dx:
            agent.x = xmax - (agent.x - xmax + 1.5*dx)
        
        if agent.y < ymin - dy:
            agent.y = ymin + (ymin - agent.y - dy)
        if agent.y > ymax - dy:
            agent.y = ymax - (agent.y - ymax + 1.5*dy)
    
    def compute_Bjvec_from_Sk(self, S_k, agent, rho=0.5):
        """ Computes the vector Bj for given agent and S_k (difference between
        target distribution and coverage distribution Fourier coefficients)
         """
        
        Nx = S_k.shape[0]
        Ny = S_k.shape[1]
        
        Lx = self.target_distribution.xmax - self.target_distribution.xmin
        Ly = self.target_distribution.ymax - self.target_distribution.ymin
        xmin = self.target_distribution.xmin
        ymin = self.target_distribution.ymin
        
        (xpos, ypos) = (agent.x, agent.y)
        (xrel, yrel) = (xpos - xmin, ypos - ymin)
        Bjx = 0.0
        Bjy = 0.0
        for kx, ky in product(range(Nx), range(Ny)):
            if self.gaussian_decay:
                k2 = kx * kx + ky * ky
                lambda_k = np.exp(-0.5 * rho ** 2 * k2)
                exit()
            else:
                lambda_k = 1.0 / pow(1.0 + kx * kx + ky * ky, 2.0)

            fkip = 1.0
            if kx != 0:
                fkip *= 0.5
            if ky != 0:
                fkip *= 0.5

            Sk_term = S_k[kx, ky]
            scale   = (lambda_k / fkip) * S_k[kx, ky]

            Bjx += scale * (-kx * pi / Lx) * sin(kx * pi * xrel / Lx) * cos(ky * pi * yrel / Ly)
            
            Bjy += scale * (-ky * pi / Ly) * cos(kx * pi * xrel / Lx) * sin(ky * pi * yrel / Ly)
   
        return Bjx, Bjy
    
    
    def plot_agents(self):
        """ Plots the current location of agents """

        xmin = self.target_distribution.xmin
        ymin = self.target_distribution.ymin
        xmax = self.target_distribution.xmax
        ymax = self.target_distribution.ymax

        # first print out agent locations
        print('Current Agent locations')
        for agent in self.agents:
            print(agent.x, agent.y)

        for agent in self.agents:
            plt.plot(agent.x, agent.y, 'ro')
        plt.axis([xmin, xmax, ymin, ymax])
        plt.show()
