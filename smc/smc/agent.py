import numpy as np
import adaptiveSampling as AS
import matplotlib.pyplot as plt
import os


""" Class for Agent """
class Agent:
    def __init__(self, init_x, init_y):
        """ Constructor for Agent class """
        self.x = init_x
        self.y = init_y

        self.x_prev = None 
        self.y_prev = None 

        self.mappingGroundTruth = []
        self.numbMeasurements = 0
        self.numbDesData = 0
        self.currentTime = 0
        self.sensingRange = 1
        

        # adaptive sampling parameters
        self.basis_center = None
        self.a_hat = None
        self.big_lamb = None
        self.lamb = None
        self.a_real = None
        self.num_basis = 5
        self.sigma = 0.1

        #self.initialize_adaptive_sampling()

    def initialize_adaptive_sampling(self, num_basis):

        # domains are the regions where the function is defined
        xDomain = 1
        yDomain = 1
        nrow = num_basis # number of basis functions in x
        ncol = num_basis # number of basis functions in y

        #self.basis_center = AS.generate_centers(xmin, xmax, ymin, ymax, nrow, ncol)
        self.basis_center = AS.generate_centers(xDomain, yDomain, nrow, ncol)


        # initialize adaptive parameters
        self.a_hat = np.zeros((len(self.basis_center), 1))

        # put a small value to avoid division by zero
        for i in range(len(self.basis_center)):
            self.a_hat[i] = 0.01

        # functions of the adaptive sampling strategy
        self.big_lamb = np.zeros((len(self.basis_center), len(self.basis_center)))
        self.lamb = np.zeros((len(self.basis_center), 1))

        self.a_real = np.zeros((len(self.basis_center), 1)) 

        return None


    def get_measurement(self, x, y):
        """Simulates a measurement for a single robot at one time instance

        Input arguments:
        robot = robot with currentlocation and ground truth measurement map
        """

        # Add noise
        sigma = 0.0
        mean = 0

        self.numbMeasurements += 1
        
        # the variable self.currentTime defines the time varying map
        newData = self.mappingGroundTruth[self.currentTime][y, x].copy()
        print(newData)
        if newData >= 0.01: 
            self.numbDesData += 1
            print('Number of measurements: ', self.numbMeasurements)
            print('Number of desired data: ', self.numbDesData)

        newData = newData + sigma*np.random.randn() + mean
        
        return newData, self.currentTime


    def createMap(self, x_sample, y_sample, newData, dataTime):
        """Creates a map of the environment for the robot
        using adaptive mapping

        Input arguments:
        robot = robot with currentlocation and ground truth measurement map
        newData = new measurement data
        dataTime = time of the measurement
        """

        # this real function is the underlying function
        # when this become time dependent, the index will be the time
        # real_func is used to get a sample in AS.adaptaion_law

        time_step = dataTime

        print('Running adaptive sampling')

        self.a_hat, self.big_lamb, self.lamb = AS.adaptaion_law(
                x_sample,       y_sample, newData,  self.a_real,
                self.a_hat,     self.basis_center,  self.sigma,
                self.big_lamb,  self.lamb,          time_step)

        print('Adaptive sampling done')
        return None


    def visualizeMap(self, xmin, xmax, ymin, ymax, timevis, method, animation_folder):
        """Visualizes the map of the environment for the robotos.path.join(animation_folder, 'coverage_NRMSE.pdf')
        """
        # this is a test to see if this function is being properly called
        xDomain = 1
        yDomain = 1

        # Domain of the parametrized function
        x_app = np.linspace(0, xDomain, 450) # x of approximated function
        y_app = np.linspace(0, yDomain, 235) # y of approximated function
        xx, yy = np.meshgrid(x_app, y_app)

        #f_hat = AS.func_hat(xx, yy, self.a_hat, self.basis_center, self.sigma)
        f_hat = AS.func_hat(xx, yy, self.a_hat, self.basis_center, self.sigma)

        # reads the underlying function
        real_func = self.mappingGroundTruth[timevis]

        plot_3d = False
        if plot_3d:

            # Create a figure and a 3D axis
            fig = plt.figure()
            ax = fig.add_subplot(121, projection='3d')

            # Plot the surface
            ax.plot_surface(xx, yy, f_hat, cmap='viridis')
            #plt.title('Function approximation')

            ax = fig.add_subplot(122, projection='3d')

            # Plot the surface
            ax.plot_surface(xx, yy, real_func, cmap='RdGy')
            #plt.title('Underlying function')
            plt.show()


        # Domain of the real function
        x = np.linspace(0, 449, 450, dtype=int)
        y = np.linspace(0, 234, 235, dtype=int)

        fig = plt.figure()

        ax = fig.add_subplot(111)

        # plot the function and its approximation
        im1 = ax.contourf(x, y, f_hat, 20, cmap='viridis')
        #plt.contourf(x, y, real_func, 20, cmap='RdGy')
        #plt.title('Function approximation')
        #plt.axis('scaled')
        #plt.colorbar()
        #plt.title('Function approximation')
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        fig.colorbar(im1, ax=ax, location='right', shrink=0.9)

        plt.savefig(os.path.join(animation_folder, method +'_app.pdf'), bbox_inches='tight')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        im2 = ax.contourf(x, y, real_func, 20, cmap='viridis')
        #plt.figure(1)
        #plt.contourf(x, y, f_hat, 20, cmap='RdGy')
        #plt.axis('scaled')
        #plt.colorbar()
        #plt.title('Underlying function')

        ax.set_yticklabels([])
        ax.set_xticklabels([])

        fig.colorbar(im2, ax=ax, location='right', shrink=0.9)

        plt.savefig(os.path.join(animation_folder, method +'_real.pdf'), bbox_inches='tight')
        #plt.savefig(method + '_real.pdf', bbox_inches='tight')

        plt.show()

        return None


class AgentTrajectories:
    def __init__(self):
        self.xs = []
        self.ys = []

    def add_point(self, x, y):
        """ adds a new point to the trajectory """
        self.xs.append(x)
        self.ys.append(y)
