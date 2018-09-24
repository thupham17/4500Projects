#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 20:39:39 2018

@author: issa18
"""
import numpy as np

#functions for Extra credit 2

def check_powerof2(k):
    if k < 0:
        raise ValueError("k must be greater than 0")
    if k == 1:
        return True
    num = 2
    while num < k:
        num *= 2
    return num == k

def powermethod2(matrix, k):
    if check_powerof2(k):
        num = 2
        while num < k:
            matrix = np.square(matrix)
            num *= 2
        print matrix
        return matrix
    else:
        raise ValueError("k must be a power of 2")