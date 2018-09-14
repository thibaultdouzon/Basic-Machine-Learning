import numpy as np
from matplotlib import pyplot as plt

def plot_disc(w, color='k'):
    if abs(w[1])>abs(w[2]):
        yr = np.arange(-6,10,1)
        xr = -(w[2]*yr+w[0])/w[1]
        valid  = (xr>-10) & (xr<10)
        plt.plot(xr[valid], yr[valid], color)
    else:
        xr = np.arange(-4,10,1)
        yr = -(w[1]*xr+w[0])/w[2]
        valid  = (yr>-10) & (yr<10)
        plt.plot(xr[valid], yr[valid],color)
def sigma(a):
    return 1./(1.+np.exp(-a))

def plot_heat(w):
    """ 
    Plot a weight vector w = [w_0,w_1,w_2] as a colour map
    """
    xx,yy = np.mgrid[-10:10:.1,-10:10:.1]
    p = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            p[i,j] = sigma(w.dot(np.array([1., xx[i,j], yy[i,j]])))
    plt.pcolor(xx,yy,p, cmap='seismic')
    plt.xlim([-10,10])
    plt.ylim([-10,10])

def part_a():
    w=np.array([1,2,1])
    x1=np.array([2,3])
    x2=np.array([-2,0])

    plot_disc(w)
    plot_heat(w)
    plt.plot(*x1, 'Xb', *x2, 'or')

    plt.legend(['discriminant', 'point(2,3)', 'point(-2,0)'])
    plt.draw()

    print(f'x1 : {w.dot(np.concatenate(([1],x1)))}, x2 : {w.dot(np.concatenate(([1],x2)))}')
    plt.show()

def part_b():
    def batch(w, x, c):
        eta = 0.5
        w_new = w + eta*np.sum([xi*ci for xi, ci in zip(x, c)])
        return w_new
    w = np.array([1,2,1])
    x = np.array([[2,3],
                [-2,0]])
    c = np.array([-1, 1])
    w_new = batch(w,x,c)
    print(w_new)

    plot_disc(w)
    plot_disc(w_new, 'g')
    plt.plot(*x[0], 'Xb', *x[1], 'or')
    plot_heat(w_new)
    plt.legend(['discriminant w', 'discriminant w_new', 'point(2,3)', 'point(-2,0)'])

    plt.draw()

    print(f'x1 : {w_new.dot(np.concatenate(([1],x[0])))}, x2 : {w_new.dot(np.concatenate(([1],x[1])))}')


    plt.show()
    pass

def part_d():
    def stochastic(w, x, c):
        eta=0.6
        for xi, ci in zip(x, c):
            xi = np.concatenate(([1], xi))
            w_new = w + eta*xi*ci
            w=w_new
        return w_new
        pass
    
    w = np.array([1,2,1])
    x = np.array([[-2,0],
                  [2,3]])
    c = np.array([1, -1])
    w_new = stochastic(w,x,c)
    print(w_new)

    plot_disc(w)
    plot_disc(w_new, 'g')
    plt.plot(*x[1], 'Xb', *x[0], 'or')
    plot_heat(w_new)
    plt.legend(['discriminant w', 'discriminant w_new', 'point(2,3)', 'point(-2,0)'])

    plt.draw()

    print(f'x1 : {w_new.dot(np.concatenate(([1],x[1])))}, x2 : {w_new.dot(np.concatenate(([1],x[0])))}')
    plt.show()
    pass

def main():
    # part_a()
    part_b()
    part_d()
    pass

if __name__ == '__main__':
    main()