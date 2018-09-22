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

def runpower(M, n, eig, v):
    #Params
    oldnormw = 0
    tolerance = 1e-06
    convergence = False

    T = 1 #number of iterations
    for t in range(T):
        if t == 0:
            w0 = np.random.rand(n).reshape(-1,1)
            #w0 = M@w/np.linalg.norm(M@w)
            #print(w0[0:10,:])
            #print(M[0:10,0:10])
            print('M1',M[0:5,0:5])
            M = M - eig*v@v.reshape(-1,1)
            print('M2',M[0:5,0:5])
            w = w0-((v@w0)*(v).reshape(-1,1))
            normw = np.linalg.norm(M@w)
            w = M@w/normw
            #print(normw)
            #print(w[0:10])
        else:
            normw = np.linalg.norm(M@w)
            #print(normw)
            w = M@w/normw

        #print('t: ',t,'normw: ',normw)

        if np.abs(normw - oldnormw)/normw < tolerance:
            convergence = True
            #print (t,'breaking')
            break
        oldnormw = normw
    #if convergence == False:
    #    raise error("Did not converge.")

    #calculate eigen using Raleigh quotient
    eig = (w.T@M@w)/(w.T@w)
    #print(eig)
    return w.T,eig

def eigen(M, n, tol):
    print('i=0')
    vector = np.zeros((M.shape[1],100000))
    eig = np.zeros(100000)
    vector[:,0], eig[0] = runpower(M, n, eig[0], vector[:,0])
    i = 0
    while (eig[i]/eig[0] >= tol):
        print('i=',i)
        if i == 1:
            break
        vector[:,i+1], eig[i+1] = runpower(M, n, eig[i], vector[:,i])
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
