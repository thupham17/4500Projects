'''
@Project 1
@author: Tarik Brahim, Sion Chun, Thu Pham, Issa Abdul Rahman, Karim Yacoub
@Description: This module finds the eigenvalues of a matrix given the variance
@Created on Sep 20, 2018
'''

import numpy as np
import sys
import time

'''class NoConvergence(Error):
   Raised when power method does not converge.
   sys.exit("Power method did not converge.")'''

def inputfile(filename):
    '''Reads the input file'''

    try:
        f = open(filename, 'r')
    except IOError:
        print ("Cannot open file \'{0}\'\n".format(filename))
        sys.exit("bye")

    # read data
    data = f.readlines()
    f.close()

    # Parse first line
    line0 = data[0].split()
    if len(line0) == 0:
        sys.exit('Empty first line.')


    if filename == 'russell_cov.txt':
        n = int(line0[1])
        matrix = np.zeros((n,n))
        line1 = data[1].split()
        #should check that line1[0] is the string 'matrix'
        for i in range(n):
            theline = data[i+2].split()
            for j in range(n):
                valueij = float(theline[j])
                matrix[i][j] = valueij

    elif filename == 'missing.dat':
        n = int(line0[1])
        print ("n = ", n)
        m = int(line0[3])
        print("m = ", m)
        matrix = np.zeros((n,m))
        for i in range(n):
            theline = data[i+1].split()
            for j in range(m):
                if theline[j] == 'NA':
                    valueij = float(-1)
                else:
                    valueij = float(theline[j])
                matrix[i][j] = valueij
    return matrix, n

def returns(matrix):
    returns = np.zeros((matrix.shape))
    for i in range(n):
        for j in range(m):
            if j == 0:
                returns[i][j] = 0
            else:
                returns[i][j] = (matrix[i][j] - matrix[i][j-1]) / matrix[i][j-1]

def fillmissing(matrix):
    n = matrix.shape[0]
    m = matrix.shape[1]
    for i in range(n):
        for j in range(m):
            if matrix[i][j] == -1:
                if j == 0:
                    next_val = 0
                    for k in range(j+1, m):
                        if matrix[i][k] != -1:
                            next_val = matrix[i][k]
                            break
                    matrix[i][j] = next_val
                elif j == m-1:
                    matrix[i][j] = matrix[i][j-1]
                else:
                    next_val = 0
                    for k in range(j+1, m):
                        if matrix[i][k] != -1:
                            next_val = matrix[i][k]
                            break
                    matrix[i][j] = (matrix[i][j-1] + next_val) / 2
    return matrix

def runpower(M,n,w):
    '''Runs the power method with normalization.'''
    T = 10000 #number of iterations
    tol = 1e-06
    convergence = False
    oldnormw = 0
    for t in range(T):
        normw = np.linalg.norm(M@w)
        w = M@w/normw

        #convergence
        if np.abs(normw - oldnormw)/normw < tol:
            convergence = True
            break
        oldnormw = normw
    #if convergence == False:
    #    raise NoConvergence

    #Calculate eigen value using Raleigh quotient
    eig = (w.T@M@w)/(w.T@w)
    return eig, np.ravel(w)

def runpower2(M,n,w):
    '''Runs the power method with no normalization and k is a power of 2.'''
    T = 10000 #number of iterations
    tol = 1e-06
    convergence = False
    oldnormw = np.linalg.norm(w/w[0])
    for t in range(10):
        w = np.linalg.matrix_power(M,2**(t))@w
        normw = np.linalg.norm(w/w[0])
        #convergence
        if np.abs(normw - oldnormw)/normw < tol:
            convergence = True
            #print('Convergence')
            break
        oldnormw = normw
    #if convergence == False:
    #    raise NoConvergence

    #Calculate eigen value using Raleigh quotient
    w=w/w[0]
    eig = (w.T@M@w)/(w.T@w)
    return eig, np.ravel(w)


def eigen(M, n, tol, power):
    vector = np.zeros((M.shape[1],100000))
    eig = np.zeros(100000)
    w0 = np.random.rand(n)
    eig[0], vector[:,0] = power(M,n,w0)
    i = 0
    while (eig[i]/eig[0] >= tol):
        M= M - eig[i]*vector[:,i].reshape(-1,1)@np.array([vector[:,i]])
        w = np.random.rand(n)
        w0 = w.reshape(-1,1) - np.array([vector[:,i]])@w.reshape(-1,1)*vector[:,i].reshape(-1,1)
        eig[i+1], vector[:,i+1] = power(M,n,w0)
        i = i+1

    return vector[:,0:i+1], eig[0:i+1]

if __name__ == '__main__':
    
    # Question 1
    M, n = inputfile()

    start = time.clock()
    eigv1, eig1 = eigen(M,n,0.5,runpower)
    end = time.clock()
    print('Question 1 eigenvector:\n{0}'.format(eigv1))
    print('Question 1 eigenvalues:{0}'.format(eig1))

    p1time = end-start
    print ('Time: {}s. Number of eigenvalues:{}'.format(p1time,len(eig1)))


    #Check eigen values with numpy function
    npeig1,npeigv1 = np.linalg.eigh(M)


    # Question 2
    M, n = inputfile('missing.dat')
    M = fillmissing(M)
    cov = np.cov(M)
    start = time.clock()
    eigv2, eig2 = eigen(cov,cov.shape[0],0.5,runpower)
    end = time.clock()
    print('eigenvalues: ', eig)
    #print('eigenvector: ', eigv)

    p2time = end-start
    print ('Question 2 time: {}s. Number of eigenvalues:{}'.format(p2time,len(eig2)))
    print('Question 2 eigenvalues:{0}'.format(eig1)) #Q2 eig


    # Question 3
    eig3 = {}
    days = [2,50,100,200,250,300,400,450,504]
    for i in days:
        cov = np.cov(M[:,0:i])
        eigv_tmp,eig[i] = eigen(cov,cov.shape[0],0.7,runpower)
    for j in eig3:
        print('Days: {0}, Eigenvalues:{1}'.format(j,eig[j]))

    #Question 4
    start = time.clock()
    eigv4, eig4 = eigen(cov,n,0.3,runpower2)
    end = time.clock()
    print('eigenvalues: ', eig4)
    p2time = end-start
    print ('Extra credit time: {}s. Number of eigenvalues:{}'.format(p2time,len(eig4)))
