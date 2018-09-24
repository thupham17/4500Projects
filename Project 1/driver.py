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

def inputfile():
    '''Reads the input file'''
    # stop the program and print an error message if no input file
    if len(sys.argv) != 2:
        sys.exit("Error: No data file.")

    filename = sys.argv[1]
    print("input", sys.argv[1])

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
    '''Reads the russell_cov input file'''
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

def eigen(M, n, tol):
    vector = np.zeros((M.shape[1],100000))
    eig = np.zeros(100000)
    w0 = np.random.rand(n)
    eig[0], vector[:,0] = runpower(M,n,w0)
    i = 0
    while (eig[i]/eig[0] >= tol):
        M= M - eig[i]*vector[:,i].reshape(-1,1)@np.array([vector[:,i]])
        w = np.random.rand(n)
        w0 = w.reshape(-1,1) - np.array([vector[:,i]])@w.reshape(-1,1)*vector[:,i].reshape(-1,1)
        eig[i+1], vector[:,i+1] = runpower(M,n,w0)
        i = i+1

    return vector[:,0:i+1], eig[0:i+1]

if __name__ == '__main__':
    # Question 1
    M, n = inputfile()
    eigv, eig = eigen(M,n,0.1)

    #Check eigen values with numpy function
    npeig,npeigv = np.linalg.eigh(M)

    print('eigenvalues: ', eig)
    print('eigenvector: ', eigv)

    # Question 2
    M, n = inputfile('missing.dat')
    M = fillmissing(M)
    cov = np.cov(M)
    print(cov[0:10,0:10])
    eigv1,eig1 = eigen(cov,cov.shape[0],0.1)
    

   
    # Question 3
    print eig1 #Q2 eig
    for i in range(2, M.len):
        cov = np.cov(M[0:i])
        eigv_tmp,eig_tmp=eigen(cov,cov.shape[0],0.1)
        if(i==M.len-1): print(eig_tmp)

