import numpy as np
from scipy.linalg import lu
from matplotlib import pyplot as plt
import random

f = lambda x: 2*x**3 - 2*x 


def compute_error(w, x, y):
    return np.sqrt(np.sum(([w.dot([x_i**j for j in range(len(w))]) for x_i in x]-y)**2)/len(x))


def gene_random_data(n=30):

    x = np.linspace(-50, 50, 1000)
    y = f(x)

    x_noise = np.array([random.random()*100-50 for _ in range(n)])
    y_noise = np.array([f(i) + (random.gauss(0,150)*200) for i in x_noise])

    return x_noise, y_noise

def poly_learn(x, y, n):
    """ This function computes the polynomial of maximal degree n that minimises 
    the squared error to fit to the data x, y
    
    """
    # Some matrices 
    # matrix of the sum of the powers of x
    X_mat = np.array([[np.sum([x_i**(r+c) for x_i in x]) for c in range(n+1)] for r in range(n+1)])
    # matrix of the sum of powers of x times y
    XY_mat = np.array([[np.sum([x_i**j * y_i for x_i, y_i in zip(x,y)])] for j in range(n+1)])
    
    # Concatenate X and Y on the columns
    X_mat = np.concatenate((X_mat, XY_mat), axis=1)

    # Compute Gauss-Jordan reduction
    pl, u = lu(X_mat, permute_l=True)

    for i in range(n+1):
        u[i,:] = u[i,:] / u[i,i]

    for i in reversed(range(n)):
        for j in reversed(range(i+1)):
            u[j,:] = u[j,:] - u[i+1,:]*u[j,i+1]

    # Result yields in last column
    return u[:,-1]



def plot_f():
    x = np.linspace(-50, 50, 1000)
    y = f(x)
    plt.plot(x,y)


def plot_poly(w):
    x = np.linspace(-50, 50, 1000)
    y = np.array([w.dot([x_i**j for j in range(len(w))]) for x_i in x])

    plt.plot(x, y, '--g')


def main():
    x,y = gene_random_data(50)
    x_test, y_test = gene_random_data(50)

    plot_f()
    plt.scatter(x,y)
    plt.scatter(x_test,y_test)
    plt.legend(['Underlying curve','Training data', 'Test Data'])
    plt.show()
    
    error_r = []
    error_test_r = []
    w_l = []
    for dim in range(20):
        w = poly_learn(x, y, dim)
        w_l += [w]
        error_r += [compute_error(w, x, y)]
        
        error_test_r += [compute_error(w, x_test, y_test)]
        print(error_test_r[-1])
    
    plt.plot(np.log(error_r))
    plt.plot(np.log(error_test_r))
    plt.legend(['Error on learning data', 'Error on test data'])
    plt.xlabel('Degree of polynomial')
    plt.ylabel('Log of the Error')
    plt.show()

    plot_poly(w_l[0])
    plt.scatter(x,y)
    plt.title('Underfitting')
    plt.show()

    plot_poly(w_l[3])
    plt.scatter(x,y)
    plt.title('Good fit')
    plt.show()

    plot_poly(poly_learn(x,y,40))
    plt.scatter(x,y)
    plt.title('Overfitting')
    plt.show()

    print(w_l[error_test_r.index(max(error_test_r))])

    pass

if __name__ == '__main__':
    main()
    plt.show()