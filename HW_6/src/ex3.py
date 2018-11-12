import numpy as np
import scipy as sc
from scipy import io
import math

from matplotlib import pyplot as plt

class Gauss:
    def __init__(self, mu, sigma):
        """Initialise a distribution with mean mu and covariance sigma
        
        Precompute and store everything that is not dependent
        on the datapont, so as to keep things efficient"""
        D = mu.size
        self.mu = mu
        if D > 1 :
            self.icov = np.linalg.inv(sigma)
            sign,ld = np.linalg.slogdet(sigma)
            if sign != 1:
                print("Sign=",sign)
        
            self.lognum = D*np.log(2*np.pi) + ld
        else:
            self.icov = 1/sigma
            self.lognum = D*np.log(2*np.pi*sigma)
    def logprob(self,x):
        """return log(p(x))"""
        d = x-self.mu
        return -.5 * (self.lognum + np.dot(np.dot(d,self.icov),d))
    
    def prob(self,x):
        """return p(x)"""
        return np.exp(self.logprob(x)) 


def predict(x, gauss_l, prior_l):
    res = []
    evidence = sum([g.prob(x)*p for g,p in zip(gauss_l, prior_l)])
    for c in [0,1]:
        likeli = gauss_l[c].prob(x)
        # print(f'Class {c}, likelihood {likeli}')
        res += [likeli*prior_l[c]/evidence]
    # print(evidence)
    return res


def part_a(d):
    A = d['A'].T
    B = d['B'].T
    
    print(f'For A: mean={A.mean(axis=0)}, cov={np.cov(A.T)}, det_cov={np.linalg.det(np.cov(A.T))}')
    print(f'For B: mean={B.mean(axis=0)}, cov={np.cov(B.T)}')
    ca = np.cov(A.T)
    cai = np.linalg.inv(ca)
    print(np.linalg.eig(ca))
    print(ca@cai)
    print(cai@ca)
    pass

def part_b(d):
    A = d['A'].T
    B = d['B'].T
    D = np.concatenate((A,B))
    d_cov = np.cov(D.T)
    w,u = np.linalg.eig(d_cov)

    print(d_cov)
    print(w,u)
    i = list(w).index(max(w))
    x = np.array([1,2,1])

    v = u[:,i]
    x1 = x.dot(v)
    print(x1)
    print(x1*v)

def part_c(d):
    A = d['A'].T
    B = d['B'].T
    D = np.concatenate((A,B))
    d_cov = np.cov(D.T)
    w,u = np.linalg.eig(d_cov)
    i = list(w).index(max(w))
    v = u[:,i]

    A1 = np.array([a.dot(v) for a in A])
    B1 = np.array([b.dot(v) for b in B])

    print(A1.mean(), np.cov(A1.T))
    print(B1.mean(), np.cov(B1.T))
    
    """
    plt.scatter(A1, np.random.rand(A1.shape[0]))
    plt.scatter(B1, np.random.rand(B1.shape[0]))
    plt.show()
    """
    x = np.array([1,2,1])
    x1 = x.dot(v)

    prior_l = [len(x)/(len(A1)+len(B1)) for x in [A1, B1]]
    print(prior_l)
    gauss_l = [Gauss(x.mean(), np.cov(x.T)) for x in [A1, B1]]
    print(x1)
    print(predict(x1, gauss_l, prior_l))

def main():
    d = io.loadmat("HW_6/assignment6_3.mat")
    # print(d['A'].T)
    # part_a(d)
    # part_b(d)
    part_c(d)
    pass


if __name__ == "__main__":
    main()

