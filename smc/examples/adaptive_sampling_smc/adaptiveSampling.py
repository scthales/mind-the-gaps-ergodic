import numpy as np
import matplotlib.pyplot as plt

def generate_centers(xDomain, yDomain, nrow, ncol):
    x = np.linspace(0, xDomain, nrow)
    y = np.linspace(0, yDomain, ncol)
    xx, yy = np.meshgrid(x, y)
    points = np.array([xx.flatten(), yy.flatten()]).transpose()

    return points


def underlying_function(x, y):
    return np.sin(2 * np.pi * x) + np.sin(2 * np.pi * y)

def loadBariumCloud(sensorPeriod):

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

    return newData, len(data)

def plotDye(image1, image2, image3):

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


def gaussian_basis(x, y, mu, sigma):
    exponent = -( 0.5 * ((x - mu[:,0]) / sigma) ** 2 + 0.5 * ((y - mu[:,1]) / sigma) ** 2)

    return 1/(2 * np.pi * sigma**2) * np.exp(exponent)

def func_hat(x, y, a_hat, basis_center, sigma):
    f = 0
    for i in range(len(a_hat)):
        # to make data compatible with the gaussian basis function
        curr_center = np.array([basis_center[i]])
        f += a_hat[i] * gaussian_basis(x, y, curr_center, sigma)

    return f


def data_weigh(t):
    """
    exponential decay function for data weighting
    """
    return 1 


def adaptaion_law(x, y, real_func, a_real, a_hat, basis_center, sigma, big_lamb, lamb, time_step):
    gamma = 0.2 # gain for adaptation
    h = 0.02 # step size for Euler's method

    scaled_x = x / 450 
    scaled_y = y / 235

    current_basis = gaussian_basis(scaled_x, scaled_y, basis_center, sigma)

    current_basis = np.array([current_basis]).transpose()

    x = h * (data_weigh(time_step) * current_basis @ current_basis.transpose())

    big_lamb = big_lamb + h * (data_weigh(time_step) * current_basis @ current_basis.transpose())

    lamb = lamb + h * (data_weigh(time_step) * current_basis * real_func)

    a_hat = a_hat - h * (gamma * (big_lamb @ a_hat - lamb))

    return a_hat, big_lamb, lamb


if __name__ == '__main__':
    # This code tests adaptive sampling

    sensorPeriod = 1
    bariumData, maxIteration = loadBariumCloud(sensorPeriod)
    # plot dye to test the data
    plotDye(bariumData[5], bariumData[30], bariumData[90])

    # domains are the regions where the function is defined
    xDomain = 1
    yDomain = 1
    nrow = 8 # number of basis functions in x
    ncol = 8 # number of basis functions in y

    basis_center = generate_centers(xDomain, yDomain, nrow, ncol)

    # Domain of the parametrized function
    x_app = np.linspace(0, xDomain, 450) # x of approximated function
    y_app = np.linspace(0, yDomain, 235) # y of approximated function
    xx, yy = np.meshgrid(x_app, y_app)

    a_real = np.zeros((len(basis_center),1)) 
    x = np.linspace(0, 449, 450, dtype=int)
    y = np.linspace(0, 234, 235, dtype=int)

    real_func = bariumData[90]
    print(real_func.shape)

    # initialize adaptive parameters
    a_hat = np.zeros((len(basis_center), 1))

    # test the adaptive sampling
    time = 50

    # functions of the adaptive sampling strategy
    big_lamb = np.zeros((len(basis_center), len(basis_center)))
    lamb = np.zeros((len(basis_center), 1))

    np.random.seed(0)
    for time_step in range(time):
        # get random sample from x
        x_sample = np.random.choice(x)
        y_sample = np.random.choice(y)
        print(x_sample, y_sample)

        a_hat, big_lamb, lamb = adaptaion_law(x_sample, y_sample, real_func, a_real, a_hat, basis_center, 0.1, big_lamb, lamb, time_step)

        
    f_hat = func_hat(xx, yy, a_hat, basis_center, 0.1)


    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    # Plot the surface
    ax.plot_surface(xx, yy, f_hat, cmap='viridis')
    plt.title('Function approximation')

    ax = fig.add_subplot(122, projection='3d')

    # Plot the surface
    ax.plot_surface(xx, yy, real_func, cmap='RdGy')
    plt.title('Underlying function')
    plt.show()

    # plot the function and its approximation
    plt.figure(0)
    plt.contourf(x, y, real_func, 20, cmap='RdGy')
    plt.axis('scaled')
    plt.colorbar()
    plt.title('Underlying function')

    plt.figure(1)
    plt.contourf(x, y, f_hat, 20, cmap='RdGy')
    plt.axis('scaled')
    plt.colorbar()
    plt.title('Function approximation')
    plt.show()
