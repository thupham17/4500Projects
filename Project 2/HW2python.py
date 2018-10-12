#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:03:19 2018

@author: issa18
"""
import numpy as np
from sympy import Eq, Symbol, solve
import sys

import time
#Step 1: Code from before
def feasible(alldata): 
    n = alldata['n']
    lower = alldata['lower']
    upper = alldata['upper']
    x = alldata['x']

    print lower, upper
    #for j in xrange(n):
    #    x[j] = lower[j]
    #x  = lower  

    x = np.copy(lower)

    sumx = np.sum(x) 

    for j in xrange(n):
        print "lower", lower, "upper",upper
        print j, 'sum', sumx, sumx + upper[j] -lower[j]
        if sumx + (upper[j] - lower[j]) >= 1.0:
            x[j] = 1.0 - sumx + lower[j]
            print 'done'
            break
        else:
            x[j] = upper[j]
            delta = upper[j] - lower[j]
            print x[j], lower[j], upper[j], delta
            sumx += upper[j] - lower[j]
        print ">>>>",j, x[j], sumx 

    print x
    alldata['x'] = x

#Reading data function from given file
def readdata(filename):
    # read data
    lines = f.readlines()
    f.close()

    line0 = lines[0].split()
    #print line0

    if len(line0) == 0:
        sys.exit("empty first line")

    n = int(line0[1])
    print "n = ", n

    lower = np.zeros(n)
    upper = np.zeros(n)
    mu = np.zeros(n)
    x = np.zeros(n)
    covariance = np.zeros((n,n))

    numlines = len(lines)
    #crude python
    linenum = 5
    while linenum <= 5 + n-1:
        line = lines[linenum-1]
        thisline = line.split()
        print thisline
        index = int(thisline[0])
        lower[index] = float(thisline[1])
        upper[index] = float(thisline[2])
        mu[index] = float(thisline[3])        
    
        linenum += 1
    linenum = n + 6
    line = lines[linenum-1]
    thisline = line.split()
    print thisline
    lambdaval = float(thisline[1])
    print "lambda = ", lambdaval
    linenum = n + 10
    while linenum <= n+10 + n-1:
        line = lines[linenum-1]
        thisline = line.split()
        print thisline
        i = linenum - n - 10
        print i
        for j in xrange(n):
            covariance[i,j] = float(thisline[j])
        linenum += 1

    print covariance

    alldata = {}
    alldata['n'] = n
    alldata['lower'] = lower
    alldata['upper'] = upper
    alldata['mu'] = mu
    alldata['covariance'] = covariance
    alldata['x'] = x

    return alldata

#the minimization function
def fx(lam, sigma, x, mu):
    second_part = np.dot(mu, x)
    first_part = 0
    left = 0
    right = 0
    i = 0
    j = 0
    for i in range(0, len(x)):
        left = left + sigma[i][i] * np.square(x[i])
        for j in range (i+1, len(x)):
            right = right + sigma[i][j] * x[i] * x[j]
    first_part = lam * (left + right * 2)
    return first_part - second_part

#the gradient
def fxd1(lam, sigma, x, mu):
    grad_vector = np.empty(len(x), float)
    i = 0
    j = 0
    for i in range(0, len(x)):
        simple = 2 * lam * sigma[i][i]*x[i] - mu[i]
        harder = 0
        for j in range (0, len(x)):
            if j != i:
                harder += sigma[i][j]*x[j]
            harder = harder * 2 * lam
        grad_vector[i] = simple + harder
    return grad_vector

#The g function with s
def gprimesolver(lam, sigma, x, y):
    s = Symbol('s')
    second_part = np.dot(mu, y)
    first_part = 0
    left = 0
    right = 0
    i = 0
    j = 0
     for i in range(0, len(x)):
        left = left + sigma[i][i] * (y[i]*x[i] + 2*s*np.square(y[i])
        for j in range (i+1, len(x)):
            right = right + sigma[i][j] * (x[i]*y[j] + x[j]*y[i] + 2*s*y[i]*y[j])
    first_part = 2 * lam * (left + right)
    eq = Eq(first_part - second_part, 0)
    sval = solve(eq)
    if sval > 1:
        return 1
    elif sval < 0:
        return 0
    else:
        return sval

#TODO: check if such vector exists or not
def linprogsolv(lower, upper, x, g):
    candidateys = list()
    m=0
    for i in range(len(x)):
        y = np.empty(len(x), float)
        for j in range(len(x)):
            if(j < m):
                y[j] = lower[j] - x[j]
            elif(j > m):
                y[j] = upper[j] - x[j]
            elif(j == m):
                y[j] = 0
        y[m] = 0 - sum(y)
        if isfeasible(x, lower, upper, y) == 1:
            candidateys.append(y)
        m += 1
    minimums = []
    for k in range(len(candidateys)):
        minimums.append(np.dot(g, candidateys[k]))
    idx = minimums.index(min(minimums))
    return candidateys[idx]


    
    
    
def isfeasible(x, lower, upper, candidate):
    counter = 0
    for j in range(len(candidate)):
        if lower[j] - x[j] <= candidate[j] and upper[j] - x[j] >= candidate[j]:
            counter += 1
    if counter == len(candidate):
        return 1
    else:
        return 0

    
#running the code
if len(sys.argv) != 2:  # the program name and the datafile
    # stop the program and print an error message
    sys.exit("usage: eigen.py datafile ")

filename = sys.argv[1]

print "input: ", sys.argv[1]

try:
    f = open(filename, 'r')
except IOError:
    print ("Cannot open file %s\n" % filename)
    sys.exit("bye")
#read data
alldata = readdata(filename)
#stuff from the reading - PLACEHOLDER
lam = 0
sigma = 0
mu = 0
lower = 0
upper = 0
#find a feasible solution
x = feasible(alldata)
#improvement phase
#Compute gk, the gradient
G = fxd1(lam, sigma, x, mu)
#Find the y vector
y = linprogsolv(lower, upper, x, G)
#Find s
s = gprimesolver(lam, sigma, x, y)
x = x + s*y


                                     
                                     
                                     

#####extra credit######
# power method from Q1
def eigen(M, n, tol, power):
    vector = np.zeros((M.shape[1],100000))
    eig = np.zeros(100000)
    w0 = np.random.rand(n)
    eig[0], vector[:,0] = power(M,n,w0)
    i = 0
    while (eig[i]/eig[0] >= tol):
        M= M - np.matmul(eig[i]*vector[:,i].reshape(-1,1),np.array([vector[:,i]]))
        w = np.random.rand(n)
        w0 = w.reshape(-1,1) - np.matmul(np.array([vector[:,i]]),w.reshape(-1,1)*vector[:,i].reshape(-1,1))
        eig[i+1], vector[:,i+1] = power(M,n,w0)
        i = i+1

    return vector[:,0:i+1], eig[0:i+1]

def runpower(M,n,w):
    T = 10000 #number of iterations
    tol = 1e-06
    convergence = False
    oldnormw = 0
    for t in range(T):
        normw = np.linalg.norm(np.matmul(M,w))
        w = np.matmul(M,w)/normw

        #convergence
        if np.abs(normw - oldnormw)/normw < tol:
            convergence = True
            break
        oldnormw = normw
    #if convergence == False:
    #    raise NoConvergence

    #Calculate eigen value using Raleigh quotient
    eig = np.matmul(np.matmul(w.T,M),w)/np.matmul(w.T,w)
    return eig, np.ravel(w)


#find eigenvectors & eigenvalues
cov = alldata['covariance']
eig_vecs, eig_vals = eigen(cov,cov.shape[0],0.5,runpower)

#check same unit length of eig vectors
for ev in eig_vecs.T:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))


#sort eigenvalues descending
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# check explained variance of eigenvalues
count = len(eig_vals)

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(count), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(count), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
plt.show()


# remove less than 5%
k=count-1
for i in range(len(var_exp)):
    if(var_exp[i]<5):
        k=i
        break

# choose k largest eigenvalues to construct projection matrix
num = len(eig_vecs)
matrix_proj = np.ndarray(shape = (num, 0))

for i in range(k+1):
    matrix_proj = np.hstack((matrix_proj, eig_pairs[i][1].reshape(num,1)))
print('Matrix proj:\n', matrix_proj)                        
                                     
                                     
