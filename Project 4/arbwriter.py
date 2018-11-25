def writelp(lpfilename, prices, deviations, numsec, numscen):
    try:
        lpfile = open(lpfilename, 'w') # opens the file
    except IOError:
        print("Cannot open LP file %s for writing\n" % lpfilename)
        return 1

    print "now writing LP to file", lpfilename
    lpfile.write("Minimize ")

    j = 0
    #Objective
    while j <= numsec:
        if prices[0][j] >= 0:
            lpfile.write("+ ")
            lpfile.write(str(prices[0][j]) + " x" + str(j)+" ")
        j += 1
    lpfile.write("\nSubject to\n")

    k = 1
    #dual constraints
    while k <= numscen:
        lpfile.write("A" + str(k) +": ")
        j = 0
        while j <= numsec:
            if prices[k][j] >= 0:
                lpfile.write("+ ")
                lpfile.write(str(prices[k][j]) + " u" + str(k) + str(j) + " -")
                lpfile.write(str(deviations[k][j]) + " u" + str(k) + str(j) + " -")
                lpfile.write(str(prices[k][j]) + " v" + str(k) + str(j) + " -")
                lpfile.write(str(deviations[k][j]) + " v" + str(k) + str(j) + " ")
            j += 1
        lpfile.write(">= 0\n")
        k += 1

    k = 1
    while k <= numscen:
        j = 0
        while j <= numsec:
            lpfile.write("B" + str(k) + str(j) +": ")
            lpfile.write("u" + str(k) + str(j) + " -")
            lpfile.write(" v" + str(k) + str(j) + " -")
            lpfile.write(" x" + str(j) + " ")
            lpfile.write("= 0\n")
            j += 1
        k += 1

    lpfile.write("Bounds\n")
    j = 0
    while j <= numsec:
        lpfile.write("-1 <= x" + str(j) + " <= 1\n")
        k = 1
        while k <= numscen:
            lpfile.write("u" + str(k) + str(j) + " >= 0\n")
            lpfile.write("v" + str(k) + str(j) + " >= 0\n")
            k += 1
        j += 1
    lpfile.write("End\n")

    print "closing lp file"
    lpfile.close()

    return 0
    
if __name__ == '__main__':
    test =
