'''
@Project 1
@author: Tarik Brahim, Sion Chun, Thu Pham, Issa Abdul Rahman, Karim Yacoub
@Description: This module finds the eigenvalues of a matrix given the variance
@Created on Sep 20, 2018
'''

import numpy as np
import sys
import time

def inputfile():
    # stop the program and print an error message if no input file
    if len(sys.argv) != 2:
        sys.exit("Error: No data file.")

    filename = sys.argv[1]
    print("input", sys.argv[1])

    try:
        f = open(filename, 'r')
    except IOError:
        print ("Cannot open file %s\n" % filename)
        sys.exit("bye")

    # read data
    data = f.readlines()
    f.close()

    line0 = data[0].split()
    #print line0

    if len(line0) == 0:
        sys.exit("empty first line")

    n = int(line0[1])
    print ('n = ', n)

    matrix = np.zeros((n,n))

    line1 = data[1].split()
    #should check that line1[0] is the string 'matrix'
    for i in range(n):
        #read line i + 2
        theline = data[i+2].split()
        #print i, " -> ", theline
        for j in range(n):
            valueij = float(theline[j])
            #print i, j, numberij
            matrix[i][j] = valueij

    return matrix, n

def runpower1(matrix, n):
    v = np.zeros(n)
    w = np.zeros(n)

    for j in range(n):
        v[j] = np.random.uniform(0,1)

    #print 'matrix', matrix
    #print 'v', v
    T = 10000 #number of iterations
    tolerance = 1e-06
    oldnormw = 0
    for t in range(T):
        w = matrix.dot(v)
        #print 't', t, 'w',w
        normw = (np.inner(w,w))**.5

        v = w/normw
        #print 't',t,'v',v

        #print 't',t,'normw',normw, 'old', oldnormw
        if np.abs(normw - oldnormw)/normw < tolerance:
            #print ' breaking'
            break
        oldnormw = normw
    #comment: if t reaches T-1 the algorithm has not converged to tolerance
    # within T iterations.  The function should return an error code in that
    # case
    eig = (v.T@M@v)/(v.T@v)
    return eig, v

def runpower2(matrix, n,v):
    #print(v)
    #print 'matrix', matrix
    #print 'v', v
    T = 10000 #number of iterations
    tolerance = 1e-06
    oldnormw = 0
    for t in range(T):
        w = matrix.dot(v)
        #print(w)
        #print 't', t, 'w',w
        normw = np.linalg.norm(w)

        v = w/normw
        #print 't',t,'v',v

        #print 't',t,'normw',normw, 'old', oldnormw
        if np.abs(normw - oldnormw)/normw < tolerance:
            #print ' breaking'
            break
        oldnormw = normw
    #comment: if t reaches T-1 the algorithm has not converged to tolerance
    # within T iterations.  The function should return an error code in that
    # case
    eig = (v.T@M@v)/(v.T@v)
    return eig, v

def eigen(M, n, tol):
    vector = np.zeros((M.shape[1],100000))
    eig = np.zeros(100000)
    eig[0], vector[:,0] = runpower1(M,n)
    i = 0
    while (eig[i]/eig[0] >= tol):
        print('i=',i)
        if i == 1:
            break
        M1= M - eig*eigv.reshape(-1,1)@np.array([eigv])
        w = np.zeros(n)
        for j in range(n):
            w[j] = np.random.uniform(0,1)
        w0 = w.reshape(-1,1) - np.array([eigv])@w.reshape(-1,1)*eigv.reshape(-1,1)
        eig[i+1], vector[:,i+1] = runpower2(M1,n, w0)
        i = i+1

    return vector, eig

if __name__ == '__main__':
    M, n = inputfile()
    #print('M',M[0:10,0:10])
    #vector, eig = eigen(M,n,0.7)
    #print(eig)
    eig,eigv = np.linalg.eigh(M)
    print(eig[::-1][0:10])
    print(eigv[::-1][0:10])
    #seig = np.sort(eig)[::-1]
    #print(seig[0:20])
    '''start = time.clock()

    numcopies = 1000
    for copy in xrange(numcopies):
        runpower(matrix, n)

    end = time.clock()

    print ' ave cpu time = ', (end-start)/numcopies'''
