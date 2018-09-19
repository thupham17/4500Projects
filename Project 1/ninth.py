#!/usr/bin/python

import numpy as np
import sys

import time


def breakexit(foo):
    stuff = raw_input("("+foo+") break> ")
    if stuff == 'x' or stuff == 'q':
        sys.exit("bye")

def runpower(matrix, n):
    #get initial vector

    v = np.zeros(n)
    w = np.zeros(n)

    for j in xrange(n):
        v[j] = np.random.uniform(0,1)

    #print 'matrix', matrix
    #print 'v', v
    T = 10000 #number of iterations
    tolerance = 1e-06
    oldnormw = 0
    for t in xrange(T):
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

#########################main

if len(sys.argv) != 2:  # the program name and the datafile
    # stop the program and print an error message
    sys.exit("usage: eigen.py datafile ")

filename = sys.argv[1]

print "input", sys.argv[1]

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
print "n = ", n


matrix = np.zeros((n,n))

line1 = data[1].split()
#should check that line1[0] is the string 'matrix'
for i in xrange(n):
    #read line i + 2
    theline = data[i+2].split()
    #print i, " -> ", theline
    for j in xrange(n):
        valueij = float(theline[j])
        #print i, j, numberij
        matrix[i][j] = valueij

breakexit('run algo?')

start = time.clock()

numcopies = 1000
for copy in xrange(numcopies):
    runpower(matrix, n)

end = time.clock()

print ' ave cpu time = ', (end-start)/numcopies
