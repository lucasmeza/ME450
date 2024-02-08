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
       but a Reuss model can be used if HT=False.'''

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

def shortFiberStiffness(Em, Ef, V, s, vm):
    '''Calculates the stiffness of a short fiber composite using the modified shear lag model.
    Inputs are stiffness of fiber and matrix (Ef and Em), Volume fraction of fiber, fiber
    aspect ratio (s) and matrix Poisson's ratio vm'''
    n = np.sqrt(2*Em/(Ef*(1+vm)*np.log(1/V)))
    a = np.cosh(n*s)
    Emprime = (Ef*(1-(1/a)) + Em)
    angle = n*s*180/(np.pi)
    E = V*Ef*(1-((Ef-Emprime)*np.tanh(n*s))/(Ef*n*s)) + (1-V)*Em
    return E
#

#%% Rotation Functions
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

#%% Laminate Functions
def laminateStiffness(C, thetas, ts):
    '''Return the stiffness of a laminate given a single input C aligned with the
    fiber orientation of the base material, an array of angles corresponding to the
    laminate stack and an array of thicknesses ts corresponding to each lamina.'''
    return sum([rotateStiffness(C,theta)*t for theta,t in zip(thetas,ts)])/sum(ts)
#

def laminateCompliance(CLam):
    return inv(CLam)
#

def isLaminateBalanced(S, plot):
    '''Determine whether a laminate is balanced by inputting a compliance matrix S.
    Plot is True or False and will plot the tension-shear interaction ratios'''

    # Define angles between 0-90 degrees
    ths = [x*pi/180 for x in range(91)]

    # Find tension-shear interaction ratios
    nxyx = []; nxyy = []
    for th in ths:
      SLR = rotateCompliance(S, th)
      nxyx += [SLR[0][2]*SLR[0][0]]
      nxyy += [SLR[1][2]*SLR[1][1]]

    # Plot the results
    if plot:
        fig,axs = plt.subplots(nrows=2, ncols=1)
        axs[0].plot(ths, nxyx)
        axs[1].plot(ths, nxyy)
        axs[1].set_xlabel(r'Angle $\theta$')
        axs[0].set_ylabel(r'$\eta_{xyx}$')
        axs[1].set_ylabel(r'$\eta_{xyy}$')

    # Return balanced or not
    return all(x==0 for x in nxyx) and all(x==0 for x in nxyy)
#

#%% Temperature Functions
def Thermal_strain(e1,e2,g12, alphm, alphf, Em, Ef, vm, vf, f, delT):
    alph11 = (alphf*f*Ef + alphm*(1-f)*Em)/(f*Ef + (1-f)*Em)
    alph22 = alphf*f*(1+vf) + alphm*(1-f)*(1+vm) - alph11*(f*vf + (1-f)*vm)
    strain = np.zeros((3,1))
    strain[0,0] = e1-alph11*delT
    strain[1,0] = e2-alph22*delT
    strain[2,0] = g12
    return strain
#
