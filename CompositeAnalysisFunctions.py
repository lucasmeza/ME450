from math import sin,cos
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#%% Constitutive Equations
def eta(P1, P2, xi):
    '''Function used in the Halpin-Tsai relationships.'''
    return ((P1/P2)-1)/((P1/P2)+xi)
#
