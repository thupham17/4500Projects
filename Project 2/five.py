#!/usr/bin/python

import numpy as np
import sys

import time

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


def breakexit(foo):
    stuff = raw_input("("+foo+") break> ")
    if stuff == 'x' or stuff == 'q':
        sys.exit("bye")

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

#########################main

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

alldata = readdata(filename)

feasible(alldata)

