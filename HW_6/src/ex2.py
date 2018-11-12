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
        self.icov = np.linalg.inv(sigma)
        sign,ld = np.linalg.slogdet(sigma)
        if sign != 1:
            print("Sign=",sign)
        
        self.lognum = D*np.log(2*np.pi) + ld
                            
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
    A = d['A']
    B = d['B']
    
    print(f'For A: mean={A.mean(axis=0)}, cov={np.cov(A.T)}')
    print(f'For B: mean={B.mean(axis=0)}, cov={np.cov(B.T)}')
    pass

def part_b(d):
    A = d['A']
    B = d['B']
    gauss_l = [Gauss(X.mean(axis=0), np.cov(X.T)) for X in [A,B]]
    prior_l = [len(l)/(len(A)+len(B)) for l in [A,B]]
    print(prior_l)

    
    x=np.array([2,1])

    print(predict(x, gauss_l, prior_l))
    plt.scatter(*A.T)
    plt.scatter(*B.T)
    plt.scatter(*x)
    plt.show()


def part_c(d):
    A = d['A']
    B = d['B']
    gauss_l = [Gauss(X.mean(axis=0), np.cov(X.T)) for X in [A,B]]
    prior_l = [len(l)/(len(A)+len(B)) for l in [A,B]]

    roc = []
    for t in np.linspace(0, 1, 1000):
        conf_mat = [[0,0],[0,0]]
        
        for x in A:
            p = predict(x, gauss_l, prior_l)
            if p[1] >= t:
                conf_mat[1][0] += 1
            else:
                conf_mat[1][1] += 1
        for x in B:
            p = predict(x, gauss_l, prior_l)
            if p[1] >= t:
                conf_mat[0][0] += 1
            else:
                conf_mat[0][1] += 1

        print(np.array(conf_mat))
        tpr = 0
        if conf_mat[0][0]+conf_mat[0][1] > 0:
            tpr = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])         
        fpr = 0
        if conf_mat[1][0]+conf_mat[1][1] > 0:
            fpr = conf_mat[1][0]/(conf_mat[1][0]+conf_mat[1][1])         
        roc += [[tpr,fpr]]
    roc=np.array(roc)
    # plt.axis(aspect='equal')
    # plt.plot(roc[:,1], roc[:,0])
    # plt.xlabel('FPrate')
    # plt.ylabel('TPrate')
    # plt.show()
    roc = np.array(list(reversed(roc)))
    roc = np.array([roc[:,1], roc[:,0]]).T
    print(roc)

    area = sum([(roc[i+1,0]-roc[i,0])*(roc[i+1,1]+roc[i,1])/2 for i in range(len(roc)-1)])
    print(area)



def main():
    d = io.loadmat("HW_6/assignment6_2.mat")
    print(d.keys())
    part_a(d)
    # part_b(d)
    # part_c(d)
    pass


if __name__ == "__main__":
    main()

