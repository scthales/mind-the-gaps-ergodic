import sys
import time
sys.path.append('../../')

import random
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os
import numpy as np
from functools import partial
from itertools import product

from smc.probbayes import AgentEnvironment
from smc.probdist import ProbDist
from smc.agent import Agent
from smc.dynamic_smc import DynamicSMC

#Adaptive control
import adaptiveSampling as AS

# %matplotlib inline

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=24)


def plotDye(image1, image2, image3):
    """Plot the dye at three different time steps

    Input arguments:
    image1 = dye at specific timestep
    image2 = dye at specific timestep
    image3 = dye at specific timestep
    path = savePath
    """

    fig, ax = plt.subplots(1,3,figsize=(18, 6))

    title = 'DyePlot'

    im = ax[0].imshow(image1, origin='lower')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')

    im = ax[1].imshow(image2, origin='lower')   
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')

    im = ax[2].imshow(image3, origin='lower')      
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')

    plt.show()
    #fig.savefig(path + title + '.png')
    
    plt.close(fig)


def loadBariumCloud(sensorPeriod):
    """Gets data from the barium cloud

    Input arguments:
    sensorPeriod = sampling time
    """

    loadPath = '../../Data/BariumCloudImages/'

    npzFile = np.load(loadPath + 'BariumCloudDataSmall.npz')

    data = npzFile['data']

    newData = []
    maxIteration = int(np.round(len(data)/sensorPeriod))

    sample = 0
    for t in range(0,maxIteration):
        if (t % ((maxIteration)/len(data))) == 0:
            sample += 1
            if sample == 100:
                sample = 99
        newData.append(data[sample])

    return newData, maxIteration


def load_gaussian_bimodal(sensorPeriod,weight,xmin,xmax,ymin,ymax):

    # dimensions of the grid
    Ny = 450
    Nx = 235

    n_snapshots = 100

    maxIteration = int(np.round(n_snapshots/sensorPeriod))

    dx = float(xmax - xmin) / float(Nx)
    dy = float(ymax - ymin) / float(Ny)


    # mean and covariance of gaussian distribution

    sig_x, sig_y = (2, 2)
    cov1 = np.array([[sig_x * sig_x, 0], [0, sig_y * sig_y]])

    sig_x, sig_y = (2.2, 2.2)
    cov2 = np.array([[sig_x * sig_x, 0], [0, sig_y * sig_y]])

    # add moving centers
    newData = []
    for time_ind in range(maxIteration):
        mu1 = np.zeros((Nx, Ny))

        x1_center, y1_center = (5, -10 + 20 * time_ind/maxIteration)
        mean1 = np.array([[x1_center], [y1_center]])

        mu1 += compute_gaussian(mu1, mean1, cov1, weight, Nx, Ny, xmin, xmax, ymin, ymax, dx, dy)


        mu2 = np.zeros((Nx, Ny))
        x2_center, y2_center = (-2, 10 - 20 * time_ind/maxIteration)

        mean2 = np.array([[x2_center], [y2_center]])

        mu2 += compute_gaussian(mu2, mean2, cov2, weight, Nx, Ny, xmin, xmax, ymin, ymax, dx, dy)

        bimodal = mu1 + mu2
        newData.append(bimodal)


    # Barium data is returned as a list of 2D numpy arrays with maxIteration elements


    return newData, maxIteration


def compute_gaussian(mu, mean, cov, weight, Nx, Ny, xmin, xmax, ymin, ymax, dx, dy):

    # Adds a gaussian density of given mean and covariance and with given weight

    pdf = partial(pdf_multivariate_gauss, mu=mean, cov=cov)

    for i, j in product(range(Nx), range(Ny)):

        x = xmin + i * dx
        y = ymin + j * dy

        mu[i, j] += weight * pdf(np.array([[x], [y]]))

    return mu


def pdf_multivariate_gauss(x, mu, cov):
    """
    Calculate the multivariate normal density (pdf)

    Arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    """

    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'

    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))

    return float(part1 * np.exp(part2))


def compute_nmse(f_hat, ground_thruth):
    
        nmse = np.linalg.norm(f_hat - ground_thruth) / np.linalg.norm(ground_thruth)
    
        print('NMSE: %f' % nmse)
    
        # return root nmse
        return np.sqrt(nmse)
        #return np.sqrt(nmse)


def warp_middle(u, gamma=1.0):
    """
    u in [0,1]  ->  w in [0,1]
    gamma > 1  => faster through the middle
    gamma < 1  => slower through the middle
    gamma = 1  => linear (no warp)
    """
    u = np.clip(u, 0.0, 1.0)
    # Symmetric S-curve with tunable slope
    ug = u**gamma
    vg = (1.0 - u)**gamma
    return ug / (ug + vg)  # stays in [0,1], fixed endpoints


if __name__ == '__main__':

    # Load data from the barium cloud

    sensorPeriod = 1 
    # load data
    # format data[time], where data[time] is a 2D numpy array

    xmin, xmax = (-10.0, 10.0)
    ymin, ymax = (-10.0, 10.0)

    bariumData, maxIteration = loadBariumCloud(sensorPeriod)

    weight = 20.0

    print('Reference Data loaded')

    Nx, Ny = bariumData[0].shape

    method = "Ours" #
    comparison = False # if True, will compare with static SMC

    #coverage error for both methods
    coverage_error_all = {"Ours": [], "Non-adaptive": []}
    map_nmse_error_all = {"Ours": [], "Non-adaptive": []}

    coverage_error_ground_thruth = {"Ours": [], "Non-adaptive": []}

    if comparison:
        methods = ["Ours", "Non-adaptive"]

        animation_folder = 'dynamic_' + 'comparison' + '_nonconvex'

        method = methods[0] # 

    else:
        print('Using method:', method)

        animation_folder = 'dynamic_' + method + '_nonconvex_moving'


    # create animation folder
    if not os.path.exists(animation_folder):
        os.makedirs(animation_folder)

    plt.figure(figsize=(10, 10))

    debug = False
    if debug:
        for timeStep in range(100):
            meas = meastime = 0
            x_sample = random.choice(range(450))
            y_sample = random.choice(range(235))
            print(x_sample, y_sample)

            newData = agent.mappingGroundTruth[agent.currentTime][y_sample, x_sample].copy()
            print(newData)
            if newData > 0.0:
                meastime = agent.currentTime

            agent.createMap(x_sample, y_sample, newData, meastime)

        agent.visualizeMap(xmin, xmax, ymin, ymax, 50)
        exit()


    # Domain of the parametrized adaptive function
    x_app = np.linspace(0, 1, 450) # x of approximated function
    y_app = np.linspace(0, 1, 235) # y of approximated function
    xx, yy = np.meshgrid(x_app, y_app)

    sim_time = 200
    n_steps = 1
    dt = 1 

    random.seed(31) # 

    n_agents = 4
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2',
              '#D55E00', '#CC79A7', '#999999', '#000000', '#FF5733']

    for method in methods if comparison else [method]:

        # reset everything for the next method
        print("Resetting agent mapping")
        print("method:", method)

        prob_dist = ProbDist(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, Nx=Nx, Ny=Ny)

        # Define DynamicSMC object
        dynamic_smc = DynamicSMC(prob_dist)
        
        # add agents to coverage object
        for i in range(n_agents):
            random_state = (xmin + (xmax - xmin) * random.random(),
                            ymin + (ymax - ymin) * random.random())
            dynamic_smc.add_agent(Agent(random_state[0], random_state[1]))

            dynamic_smc.agents[i].x_prev = random_state[0]
            dynamic_smc.agents[i].y_prev = random_state[1]

            dynamic_smc.agents[i].mappingGroundTruth = bariumData
            
            dynamic_smc.agents[i].initialize_adaptive_sampling(num_basis=9)

            # sigma for the gaussian basis
            dynamic_smc.agents[i].sigma = 0.1

            dynamic_smc.agents[0].currentTime = 0 # 25

        #figure sutff

        plt.axis('equal')
        ax = plt.gca()
        ax.set_xlim(-0, 450)
        ax.set_ylim(-0, 235)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_adjustable('box')

        time.sleep(5)

        for time_ind in range(sim_time):
            print('Running Step %s of animation.' % time_ind)

            for ind, agent in enumerate(dynamic_smc.agents):

                ###################
                ## Adaptive part ##
                ###################

                # rescale x domain to 0-1, then size of the map
                x_sample = int((agent.x - xmin) / (xmax - xmin) * (agent.mappingGroundTruth[0].shape[1] - 1))
                y_sample = int((agent.y - ymin) / (ymax - ymin) * (agent.mappingGroundTruth[0].shape[0] - 1))

                meas, meastime = agent.get_measurement(x_sample, y_sample)

                if time_ind % 1 == 0:
                    agent.createMap(x_sample, y_sample, meas, meastime)

                    # update parameters of adaptive function
                    dynamic_smc.agents[0].a_hat = agent.a_hat
                    dynamic_smc.agents[0].big_lamb = agent.big_lamb
                    dynamic_smc.agents[0].lamb = agent.lamb


                if time_ind % 1 == 0:
                    nor_time = np.clip(time_ind / sim_time, 0.0, 1.0)
                    # small gamma -> slow in the middle
                    mov_time = warp_middle(nor_time, gamma=0.3)

                    agent.currentTime = int(round(99*mov_time))
                    #agent.currentTime = 25

                if time_ind % 5 == 0:
                    #print('Time: %d' % time_ind)
                    f_hat = AS.func_hat(xx, yy, agent.a_hat, agent.basis_center, agent.sigma)


            prob_dist.set_zero()

            # This needs to change with different methods
            if method == "Ours":
                prob_dist.set_prob_dist_from_array(f_hat)

            elif method == "Non-adaptive":
                grid_ones = np.ones(bariumData[0].shape)
                prob_dist.set_prob_dist_from_array(grid_ones)

            # set the ground truth for the current time step
            prob_dist.set_prob_dist_ground_thruth(bariumData[dynamic_smc.agents[0].currentTime])

            # compute the coverage error with respect to the ground truth for each method
            map_nmse_error_all[method].append(compute_nmse(f_hat, bariumData[dynamic_smc.agents[0].currentTime]))


            ########################################

            #print('Running Dynamic SMC')
            dynamic_smc.time_steps(n_steps, dt)


            ax.imshow(bariumData[dynamic_smc.agents[0].currentTime], cmap='Blues', origin='lower', vmin=0, vmax=1)


            if time_ind < 2:
                for ind, agent in enumerate(dynamic_smc.agents):
                    x_curr = int((agent.x - xmin) / (xmax - xmin) * (agent.mappingGroundTruth[0].shape[1] - 1))
                    y_curr = int((agent.y - ymin) / (ymax - ymin) * (agent.mappingGroundTruth[0].shape[0] - 1))

                    agent.x_prev = x_curr
                    agent.y_prev = y_curr

            if time_ind >= 2:

                for ind, agent in enumerate(dynamic_smc.agents):
                    x_curr = int((agent.x - xmin) / (xmax - xmin) * (agent.mappingGroundTruth[0].shape[1] - 1))
                    y_curr = int((agent.y - ymin) / (ymax - ymin) * (agent.mappingGroundTruth[0].shape[0] - 1))

                    plt.plot([agent.x_prev, x_curr], [agent.y_prev, y_curr], '-', color=colors[ind], markersize=3)

                    agent.x_prev = x_curr
                    agent.y_prev = y_curr

                    agent_pos = patches.Ellipse((x_curr, y_curr), 4.0, 4.0, facecolor='r', alpha=0.8)
                    ax.add_patch(agent_pos)

                out_fig_name = os.path.join(animation_folder,
                                            method + 'output_%d.pdf' % time_ind)

                if time_ind % 10 == 0:
                    start = time.time()
                    print('Saving figure to %s' % out_fig_name)
                    plt.savefig(out_fig_name, bbox_inches='tight')

                    print('Time to save figure: %f' % (time.time() - start))


        plt.show()

        dynamic_smc.agents[0].visualizeMap(xmin, xmax, ymin, ymax, dynamic_smc.agents[0].currentTime, method, animation_folder)


        des_data = 0
        for _, agent in enumerate(dynamic_smc.agents):
            des_data += agent.numbDesData / agent.numbMeasurements

        print("Desired data per agent:")
        print(des_data / n_agents)

        plt.show()
        plt.clf()


        coverage_error_all[method] = dynamic_smc.S_k_avg  

        coverage_error_ground_thruth[method] = dynamic_smc.S_k_gt_avg # ground truth

        print(coverage_error_all[method][-1])

    # Plot the coverage error for both methods
    if comparison:
        plt.close()
        f, ax = plt.subplots(constrained_layout=True)
        ax.plot(range(sim_time), map_nmse_error_all["Ours"], ':', linewidth=2.0, color='black', label='Ours')
        ax.plot(range(sim_time), map_nmse_error_all["Non-adaptive"], '--', linewidth=2.0, color='black', label='Non-adaptive')
        plt.xlabel('Time')
        plt.ylabel('NRMSE')

        plt.legend()
        plt.grid()

        plt.savefig(os.path.join(animation_folder, 'coverage_NRMSE.pdf'), bbox_inches='tight')

        plt.show()
