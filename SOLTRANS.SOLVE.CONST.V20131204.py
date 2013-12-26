# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:28:50 2013
This code simulates the problem in exercise 4-5 p.83 in ??? 

After experimenting, best parameter combination:
L = 100.            
v  = 4. 
D = 5.                          
dx = v*dt/(Cr=1.1)           
Pe = v*dx/D = 0.145455
D = D - v*dx/2  =  4.63636 (adjustment for numerical dispersion)
@author: tamahain
"""
import time
import numpy as np
from scipy.integrate import odeint
import matplotlib.pylab as plt

# start timing execusion
start_time = time.time()


#---------------------------------ODE system----------------------------------
  
def dCdt(C, t):
    #upwind scheme discretize PDE using differencing biased in direction 
    #determined by the sign of the characteristic velocity
    #http://en.wikipedia.org/wiki/Upwind_scheme
    #Jx = D/dx*(C[i]-C[i-1])+(V+*C[i-1]+V-*C[i])
    #where v+ = max(v,0), v-=min(v,0)
    #applying continuity eq to fluxes: dC/dt = Jx2-Jx1 results in eq used here
    Cdot = np.zeros(N)
    
    #pulse input
    if t < 36.:
      C_input = CIN
    else:
      C_input = 0
    
    #upper boundary: Type 2 Neumann (no flux)
    #diffuive flux
    Cdot[0] = (D/dx**2)*(C[1]-C[0])
    #advective flux
    Cdot[0] = Cdot[0] - (max(v,0)*(C[0] - C_input)/dx + min(v,0)*(C[1] - C[0])/dx)
    
    #middle cells
    #diffusive flux
    Cdot[1:-1] = (D/dx**2)*(C[2:]-2*C[1:-1]+C[:-2])
    #advective flux
    Cdot[1:-1] = Cdot[1:-1] - (max(v,0)*(C[1:-1] - C[:-2])/dx + min(v,0)*(C[2:] - C[1:-1])/dx)
    
    #lower boundary: Type 1 Dirichlet (concentration controlled)
    #diffusive flux (upwind only)
    Cdot[-1] = -(D/dx**2)*(C[-1]-C[-2])
    #advective flux (downwind & upwind)
    Cdot[-1] = Cdot[-1] - (max(v,0)*(C[-1] - C[-2])/dx + min(v,0)*(0 - C[-1])/dx)
    return Cdot
    
#---------------------------------Model parameters----------------------------

'''
Physical and chemical properties of the simulated soil column
L: Depth of transport domain in cm
dx: segment thickness in cm
N: No. of segments
x: coordinates of segment centres in [cm]

Kd: solid-liquid distribution coefficient in [m3/kg]
v: soil porewater velocity (Darcy flux/teta) in [cm/h]
DL: Hydrodynamic dispersion coefficient [m2/h] adjusted for numerical dispersion 
    effect

CIN: Influx conc. in [mg/cm3], converted from mg/l to mg/cm3
'''
L = 100.           

#Simulation time (start, final &  # time step) 
t0   = 0
tf = float(raw_input("enter simulation time in hours: "))
dt   = 0.05      
tspan = np.arange(t0, tf + dt, dt)   

#physical/chemical parameters  
v  = 1.
D = .5                          
Kd = {'Cl': 0,'Se':0.2, 'I':0.01} 

#for numerical stability (Courant No. < 1.0)
dx = v*dt/1.1           
N = int(L/dx)          
x = np.arange(0., L + L/N, L/N) 

#Peclet number: dominant transport process (advection vs dispersion)
Pe = v*dx/D
print 'Peclet number: %g' %Pe

#Courant number Cr: for numerical stability Cr = 1.0
Cr = v*dt/dx
print 'Courant number: %g' %Cr

#adjust physical dispersion to numerical dispersion
D = D - v*dx/2
print 'adjusted D: %g' %D

#---------------------------------IC & BC-------------------------------------                   

C0 = np.zeros(N)
CIN = float(raw_input("enter influx conc. in mg/l: "))/1000.  

#---------------------------------calling the solver--------------------------

#odeint solver
C = odeint(dCdt, C0, tspan)

#---------------------------------postprocessing------------------------------

#plotting
fig = plt.figure()

ax1 = plt.subplot(121)
ax1.plot(tspan, C[:,N-1]/CIN, 'b-')
plt.title('$Breakthrough \ curve$')
plt.xlabel('$time \ [hours]$')
plt.ylabel('$C/C0 \ [-]$')
plt.grid(True, 'major', 'both')

ax2 = plt.subplot(122)
ax2.plot(C[(20*24),:],x[1:], 'b-', label = '$t = 24 \ hr$')
ax2.plot(C[(20*36),:],x[1:], 'g-', label = '$t = 36 \ hr$') 
ax2.plot(C[(20*48),:],x[1:], 'r-', label = '$t = 48 \ hr$') 
ax2.plot(C[(20*72),:],x[1:], 'c-', label = '$t = 72 \ hr$') 
ax2.plot(C[(20*120),:],x[1:], 'k-', label = '$t = 120 \ hr$') 
plt.xlabel('$Concentration \ [mg/cm^3]$')
plt.ylabel('$depth \ [cm]$')
plt.grid(True, 'major', 'both')
plt.legend(loc='best')
locs = np.arange(x[0],x[-1]+10,10)
plt.yticks(locs)
plt.ylim(x[1], x[-1])
plt.gca().invert_yaxis()

#printing execusion time
print time.time() - start_time, "seconds"

#show graphs
plt.show()
#=============================================================================

