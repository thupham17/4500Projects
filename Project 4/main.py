#!/usr/bin/python

import sys
from arbwriter import writelp
from mysolver import lpsolver


if len(sys.argv) != 4:
    sys.exit("usage: main.py datafilename datafilename lpfilename\n")

#now open and read data file with price information

try:
    datafile = open(sys.argv[1], 'r') # opens the data file
except IOError:
    sys.exit("Cannot open file %s\n" % sys.argv[1])


lines = datafile.readlines();
datafile.close()

#print lines[0]
firstline = lines[0].split()
#print "first line is", firstline

numsec = int(firstline[1])
numscen = int(firstline[3])
r = float(firstline[5])
print "\n"
print "number of securities:", numsec,"number of scenarios", numscen,"r",r
print "\n"

#allocate prices as two-dim array

#p = [1 + numscen][1 + numsec]
p = [[0]*(numsec + 1) for _ in range(numscen + 1)]
i = 0
# line k+1 has scenario k (0 = today)
while i <= numscen:

    thisline = lines[i + 1].split()
    # should check that the line contains numsec + 1 words

    p[i][0] = 1 + r*(i != 0) # handles the price of cash

    j = 1

    while j <= numsec:
        value = float(thisline[j])
        p[i][j] = value
        # print ">>", "k",k, "j",j, k*(1 + numsec) + j
        j += 1

    i += 1

print(p)
print "\n"


#open and read file with deviation information
try:
    datafile = open(sys.argv[2], 'r') # opens the data file
except IOError:
    sys.exit("Cannot open file %s\n" % sys.argv[2])

lines = datafile.readlines();
datafile.close()

l = 0
while(lines[l].split()[0] != "END"):
    l += 1

info_line = lines[l - 1].split()
securities = int(info_line[0])
scenarios = int(info_line[1])

dev = [[0]*(securities + 1) for _ in range(scenarios)]

#i = 0
#j = 0
m = 0

while (m < (securities + 1)*scenarios):

    thisline = lines[m].split()

    deviation = float(thisline[2])

    i = int(thisline[1]) - 1
    j = int(thisline[0])

    dev[i][j] = deviation

    m += 1

print(dev)




#now write LP file, now done in a separate function (should read data this way, as well)

lpwritecode = writelp(sys.argv[3], p, numsec, numscen)

print "wrote LP to file", sys.argv[3],"with code", lpwritecode

now solve lp

lpsolvecode = lpsolver(sys.argv[3], "test.log")

print "solved LP at", sys.argv[3],"with code", lpsolvecode
