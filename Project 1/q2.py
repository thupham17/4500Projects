#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:37:00 2018

@author: issa18
"""
import sys
import numpy as np
#
if len(sys.argv) != 2:  # the program name and the datafile
    # stop the program and print an error message
    sys.exit("usage: eigen.py datafile ")

filename = sys.argv[1]

print "input" + sys.argv[1]

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
m = int(line0[3])
print "m = ", m


matrix = np.zeros((n,m))

#line1 = data[1].split()
#should check that line1[0] is the string 'matrix'
#-1 to exclude last line
for i in xrange(n):
    #read line i + 2
    theline = data[i+1].split()
    #print i, " -> ", theline
    for j in xrange(m):
        if theline[j] == 'NA':
            valueij = float(-1)
        else:
            valueij = float(theline[j])
        #print i, j, numberij
        matrix[i][j] = valueij

#handling 'NA': take average of day before and day after
#convert all non-NA to float
"""for i in xrange(n):
    for j in xrange(m):
        if matrix[i][j] != 'NA':
            matrix[i][j] = float(matrix[i][j])
"""
#convert NA to avg of day before and after
#TODO: might need to handle sequential NAs at end of line
for i in xrange(n):
    for j in xrange(m):
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
                
#Compute returns
returns = np.zeros((n,m))
for i in xrange(n):
    for j in xrange(m):
        if j == 0:
            returns[i][j] = 0
        else:
            returns[i][j] = (matrix[i][j] - matrix[i][j-1]) / matrix[i][j-1]

#compute covariance matrix
covar = np.zeros((n,n))
for i in xrange(n):
    for j in xrange(n):
        sum = 0
        for k in xrange(m):
            sum += returns[i][k]*returns[j][k]
        covar[i][j] = sum/(m-1)

print(covar)

       