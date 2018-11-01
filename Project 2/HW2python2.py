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

    #print (lower, upper)
    #for j in xrange(n):
    #    x[j] = lower[j]
    #x  = lower

    x = np.copy(lower)

    sumx = np.sum(x)

    for j in range(n):
        #print ("lower", lower, "upper",upper)
        #print (j, 'sum', sumx, sumx + upper[j] -lower[j])
        if sumx + (upper[j] - lower[j]) >= 1.0:
            x[j] = 1.0 - sumx + lower[j]
            #print ('done')
            break
        else:
            x[j] = upper[j]
            delta = upper[j] - lower[j]
            #print (x[j], lower[j], upper[j], delta)
            sumx += upper[j] - lower[j]
        #print (">>>>",j, x[j], sumx)

    #print (x)
    alldata['x'] = x
    return x

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
    #print ("n = ", n)

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
        #print (thisline)
        index = int(thisline[0])
        lower[index] = float(thisline[1])
        upper[index] = float(thisline[2])
        mu[index] = float(thisline[3])

        linenum += 1
    linenum = n + 6
    line = lines[linenum-1]
    thisline = line.split()
    #print (thisline)
    lambdaval = float(thisline[1])
    #print ("lambda = ", lambdaval)
    linenum = n + 10
    while linenum <= n+10 + n-1:
        line = lines[linenum-1]
        thisline = line.split()
        #print (thisline)
        i = linenum - n - 10
        #print (i)
        for j in range(n):
            covariance[i,j] = float(thisline[j])
        linenum += 1

    #print (covariance)

    alldata = {}
    alldata['n'] = n
    alldata['lower'] = lower
    alldata['upper'] = upper
    alldata['mu'] = mu
    alldata['covariance'] = covariance
    alldata['x'] = x
    alldata['lambda'] = lambdaval

    return alldata

#the minimization function


def fx(lam, sigma, x, mu):
    #print(x)
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

'''
def fx10(lam, sigma, x, mu):
    second_part = np.dot(mu, x)
    first_part = lam*np.matmul(np.matmul(x,sigma),x)
    return first_part - second_part

'''

#the gradient
def fxd1(lam, sigma, x, mu):
    grad_vector = np.empty(len(x), float)
    for i in range(len(x)):
        simple = 2 * lam * sigma[i][i]*x[i] - mu[i]
        harder = 0
        for j in range(len(x)):
            if j != i:
                harder = harder + (sigma[i][j]*x[j])
        harder = harder * 2 * lam
        grad_vector[i] = simple + harder
    return grad_vector

#The g function with s
def gprimesolver(lam, sigma, x, y, mu):
    s = Symbol('s')
    second_part = np.dot(mu, y)
    first_part = 0
    left = 0
    right = 0
    for i in range(0, len(x)):
        left = left + sigma[i][i] * (y[i]*x[i] + s*np.square(y[i]))
        for j in range(i+1, len(x)):
            right = right + sigma[i][j]*(x[i]*y[j] + x[j]*y[i] + 2*s*y[i]*y[j])
    first_part = 2 * lam * (left + right)
    eq = Eq(first_part - second_part, 0)
    sval = solve(eq)
    sval_out = sval[0]
    print("Gval = ",fx(lam, sigma, x+sval_out*y, mu))
    #print('sval is', sval)
    if sval_out > 1.0:
        #print('about to return 1')
        return 1
    elif sval_out < 0.0:
        return 0
    else:
        return sval_out

def linprogsolv(lower, upper, x, g):
    #print('In LinProgSolve')
    candidateys = list()
    m=0
    #sortedg = sorted(g, reverse = True)
    indexes = np.argsort(g)[::-1]
    print(indexes)
    #print('g is', g)
    #print('indexes is', indexes)
    xordered = orderit(indexes, x)
    #print('unordered x is', x)
    #print('ordered x is', xordered)
    upperoredered = orderit(indexes, upper)
    lowerordered = orderit(indexes, lower)


    for i in range(len(x)):
        print('m = ',m)
        yordered = np.empty(len(x), float)
        for j in range(len(x)):
            #print(q)
            if(j < m):
                yordered[j] = lowerordered[j] - xordered[j]
            elif(j > m):
                yordered[j] = upperoredered[j] - xordered[j]
            else:
                yordered[j] = 0
        yordered[m] = 0 - sum(yordered)
        #print(yordered)
        print('y[m] ',yordered[m])
        # print('Y is', y)
        y = unorderit(indexes, yordered)
        #print(y)
        #print('y[m] ',y[y.tolist().index(yordered[m])])
        print('y[m] ',y[indexes[m]])
        #print('y is', y)
        if isfeasible(x, lower, upper, y, indexes[m]) == 1:
            print('feasible')
            candidateys.append(y)
            #print('Y is feasible')
            #print(candidateys)
        m += 1
    #stopping condition
    if len(candidateys) == 0:
        #print('Returning 0')
        return np.zeros(len(x))
    minimums = []
    for k in range(len(candidateys)):
        minimums.append(np.dot(g, candidateys[k]))
    idx = minimums.index(min(minimums))
    #print('Idx is', idx)
    #print(candidateys)
    #print(candidateys[idx])
    return candidateys[idx]



def isfeasible(x, lower, upper, candidate,m):
    print('y[m] ',candidate[m])
    print("lower",lower[m] - x[m])
    print("upper",upper[m] - x[m])
    if (candidate[m] >= (lower[m] - x[m]) and candidate[m] <= (upper[m] - x[m])):
        return 1
    else:
        return 0

def orderit(index, thelist):
    ordered = np.empty(len(thelist), float)
    for i in range(0, len(thelist)):
        ordered[i] = thelist[index[i]]
    return ordered

def unorderit(index, thelist):
    unordered = np.empty(len(thelist), float)
    for i in range(0, len(thelist)):
        unordered[index[i]] = thelist[i]
    return unordered

#running the code
if len(sys.argv) != 2:  # the program name and the datafile
    # stop the program and print an error message
    sys.exit("usage: eigen.py datafile ")

filename = sys.argv[1]

#print ("input: ", sys.argv[1])

try:
    f = open(filename, 'r')
except IOError:
    #print ("Cannot open file %s\n" % filename)
    sys.exit("bye")
#read data
alldata = readdata(filename)
#stuff from the reading
n = alldata['n']
lam = alldata['lambda']
sigma = alldata['covariance']
mu = alldata['mu']
lower = alldata['lower']
upper = alldata['upper']
s = 0

#find a feasible solution
x = feasible(alldata)


print('The first feasible x is:', x)
#print ("X is " + str(x))
#improvement phase
#Compute gk, the gradient
G = fxd1(lam, sigma, x, mu)
#print ("G is " + str(G))
#tester = fxd1(lam, sigma, [1,1,1,1], mu)
#print ("tester is" + str(tester))
#Find the y vector

y = linprogsolv(lower, upper, x, G)


#print ("Y is " + str(y))

numberofloops = 0

while (numberofloops < 5):
    numberofloops += 1
    print('\n')
    print('The number of loops is:', numberofloops)
    print('\n')
    #print('In the Loop')
    s = gprimesolver(lam, sigma, x, y, mu)
    print('The latest x is:', x)
    print('The latest y is:', y)
    print('S is', s)
    oldobj = fx(lam, sigma, x, mu)

    x = x + s*y
    #print('x is', x)
    latestobj = fx(lam, sigma, x, mu)

    if(oldobj - latestobj < 0.0000001):
            break

    print('\n')
    print('The Latest Objective Value is', latestobj)
    #print('The corresponding x is', x)
    print('\n')

    G = fxd1(lam, sigma, x, mu)
    y = linprogsolv(lower, upper, x, G)
    #print('value of loss func is', fx(lam, sigma, x, mu))
#print('X is next')
#print(x)
print('\n')
print('\n')
print('\n')
print('\n')
print('\n The Final Solution is Below:')
print('\n')
final = fx(lam, sigma, x, mu)
print('The Objective Value is:', final)
print('\n')
print('The correspinding x is:', x)
print('\n')
#print('final is', final)
