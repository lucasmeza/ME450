from math import sin,cos
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

#%% Constitutive Equations
def eta(P1, P2, xi):
    '''Function used in the Halpin-Tsai relationships.'''
    return ((P1/P2)-1)/((P1/P2)+xi)
#

def transIsoStiffness(E1, nu1, E2, nu2, V, HT=True):
    '''Return the 2D stiffness tensor of a transversely isotropic composite made
       from materials 1 and 2. Inputs required are the stiffness and Poisson's ratio of
       both materials (E1, nu1, E2, nu2) and volume fraction V of material 1.
       Properties are calculated using Halpin-Tsai relationships by default (HT = True),
       but a Reuss model can be used if desired.'''

    # Calculate Base Material Properties
    G1 = E1/(2*(1+nu1))
    G2 = E2/(2*(1+nu2))

    # Calculate Material Properties
    Ex = E1*V + E2*(1-V) #Voigt

    if HT: #Halpin-Tsai
        Ey = E2*(1 + 2*eta(E1,E2,2)*V)/(1 - eta(E1,E2,2)*V)
        Gxy = G2*(1 + 1*eta(G1,G2,1)*V)/(1 - eta(G1,G2,1)*V)
    else: #Reuss
        Ey = 1/(V/E1+(1-V)/E2)
        Gxy = 1/(V/G1+(1-V)/G2)

    nuxy = nu1*V+nu2*(1-V) #Voigt
    nuyx = nuxy*Ey/Ex #Reciprocal

    # Create Stiffness Matrix
    C = [[Ex/(1-nuxy*nuyx),      nuyx*Ex/(1-nuxy*nuyx), 0],
         [nuyx*Ex/(1-nuxy*nuyx), Ey/(1-nuxy*nuyx),      0],
         [0,                     0,                     Gxy]]

    return np.array(C)
#

def transIsoCompliance(E1, nu1, E2, nu2, V, HT=True):
    return inv(transIsoStiffness(E1, nu1, E2, nu2, V, HT))
#

def T_matrix(theta):
    '''Matrix for stress rotation in 2D.'''
    c = cos(theta)
    s = sin(theta)
    return np.array([[c**2, s**2, 2*s*c],[s**2, c**2, -2*s*c],[-s*c, s*c, c**2-s**2]])
#

def Tp_matrix(theta):
    '''Matrix for strain rotation in 2D.'''
    c = cos(theta)
    s = sin(theta)
    return np.array([[c**2, s**2, s*c],[s**2, c**2, -s*c],[-2*s*c, 2*s*c, c**2-s**2]])
#

def rotateStress(stress, theta):
    '''Rotate 2D stress counterclockwise. Input stress as a 3x1 vector'''
    T = T_matrix(theta)
    return T.dot(stress)
#

def rotateStrain(strain, theta):
    '''Rotate 2D strain counterclockwise. Input  strain as a 3x1 vector'''
    TP = Tp_matrix(theta)
    return TP.dot(strain)
#

def rotateStiffness(C, theta):
    '''Rotate the 2D stiffness tensor C counterclockwise by angle theta.'''
    # Define rotation matrix T and T'
    T = T_matrix(theta)
    TP = Tp_matrix(theta)

    # Stress tensor rotations
    return inv(T).dot(C).dot(TP)
#

def rotateCompliance(S, theta):
    '''Rotate the 2D compliance tensor S counterclockwise by angle theta.'''
    # Define rotation matrix T and T'
    T = T_matrix(theta)
    TP = Tp_matrix(theta)

    # Stress tensor rotations
    return inv(TP).dot(S).dot(T)
#
