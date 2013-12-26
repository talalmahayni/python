# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:28:50 2013
Code not working!

The system AX = B can be solved if A is a square matrix. if A is not a asquare
matrix (such as the one that resulted from the implicit descretisation of the
ODE could not be solved using the sciypy.linalg.solve)
@author: tamahain
"""
import time
import numpy as np
from scipy.integrate import ode, odeint
import matplotlib.pylab as plt

# start timing execusion
start_time = time.time()


def rhs(t, A):
  ''' define the rate of change, or the rhs of the ode system of RN dynamics.
      the output from this function is passed to and ode integrator for solut-
      ion.
  '''
  # calculated day of year from t of the integration interval  
  t = int(t-np.trunc(t/t_dy)*t_dy) # necessary for arrays indexing 
  
  # Retardiation factor (constant with depth) as a function of soil moisture
  # & soil characteristics
  R = 1 + (1-e)*rho*Kd['I']/teta
  
  # Fractional transfer rates are calculated following the approach of
  # Klos (2008): GEMA3D-Landscape modelling for dose assessments. TCs
  # represent the fraction of the inventory in one compartment that is
  # transferred either to other compartments or out from the biosphere. 
  # leach and upwel quantify solute movement in both directions by 
  # advection and dispersion processes
  
  # Leaching by advection and diffusion rate constants    
  TC_leach_adv =  pr[:,1:]/(teta*R*dz)
  TC_leach_dif =  Dm/(teta*R*dz**2)
  TC_leach     =  TC_leach_adv + TC_leach_dif
  
  # Groundwater upwelling by advection and diffusion rate constants
  TC_upwel_adv =  cr[:,:-1]/(teta*R[:]*dz)
  TC_upwel_dif =  Dm/(teta*R*dz**2)
  TC_upwel     =  TC_upwel_adv + TC_upwel_dif
  
  # since no loss through surface via capillarity
  TC_upwel[0,:] = switch['off']*TC_upwel[0,:]
  
  # Uptake (passive) by crop rate constants  
  TC_uptk  =  uptk[:,1:]/(teta*R*dz)
  
  # Construct diagonal matrix: lower, middle and upper diagonals
  lel = TC_leach[t,:-1]*dt
  mel = - (TC_leach[t,:] + TC_upwel[t,:] + TC_uptk[t,:] + TC_dcy['I'])*dt 
  uel = TC_upwel[t,1:]*dt
  
  lel = np.diagflat(lel,-1)
  mel = np.diagflat(mel, 0)
  uel = np.diagflat(uel, 1)

  # construct the tridiagonal matrix  
  TC = lel + mel + uel
  
  # add uel vector at bottom of TC (for the crop compartment)
  TC = np.vstack((TC, TC_uptk[t,:]*dt))
  
  # add extra column to the right contains coefficients of As in the crop
  # activity balance equation. last ele of this row should be (1/dt+f): f is
  # fraction of crop biomass removed by harvesting. f should be made time
  # dependent function, a switch with value of 1 on day of harvest, and 0
  # otherwise. PAHTWAY model formulat for the harvest fraction of crop biomass
  # it is dy_hrvs -1 instead of dy_hrvs because of python 0-based indexing

  
  f_hrvs = lambda t: 1 - Bmin/B[t] if t == dy_hrvs-1 else 0
    
  TC_crop = np.append(np.zeros((TC.shape[0]-1,1)), 1 - TC_dcy['I']*dt)
  
  # to use the hstack, reshape the 1d array into 2d array
  TC_crop = TC_crop.reshape(TC.shape[0],1)
  
  # stack ele of the TC_crop_coeff 2d array to evey row of TC array    
  TC = np.hstack((TC,TC_crop))

  # two sources, infiltration (trhough irrigation) & influx of contaminated
  # groundwater 
  S[0]  = switch['off'] * Cgw*irr[t]
  S[-2] = switch['on']  * Cgw*cr[t,-1]
  S[-1] = A[-1]*(1-f_hrvs(t)*dt) # something wrong in here!!!
  
  F = np.dot(TC, A) + S
 
  return F

#-----------------------------------------------------------------------------
  
''' load hydrological data from input files. The data are outputs from hydro-
    logical simulations with the Hydrus 1D code for Sutton-Bonington site in
    the UK. A 100 cm sandy loam soil column with winter wheat crop was simula-
    ted for a period of one year assuming water initially at field capacity &
    that no irrigation is practiced. For the boundary conditions, weather data
    obtained from a weather station was used to specify atmospheric boundary
    condition at the surface and to calculate potential evapotranspiration rat
    -es (internally within Hydrus). At the bottom boundary, a static water tab
    -le was assumed. The Hydrus outputs used here are:
      teta [cm3/cm3]: moisture content per layer. This is average of two nodes
      pr [cm3/cm2/d]: percolation flux including infiltration at surface.
      cr [cm3/cm2/d]: capillary/upwelling flux including outflux through surf-
      ace.
      uptk [cm3/cm2/d]: uptake by plants. 
   pr[:,0] into soil surface not needed in transfer rates but in source term @
   the surface (inputs through irrigation).
   cr[:,0] through soil surface not needed (RN lost by volatilisation from soi
   -l surface, and volatile RN are not considered here)
   uptk[:,0] not needed.
   irr[cm3/cm2/d]: irrigation rates are stored in a csv file and used to simulate 
   spikes/inputs of activity into the soil folloiwng irrigation events 
   B [kg/m2]: crop (wheat) total biomass simulated using Sirius crop model. 
   Details of this simulation are unkown since Niel Crout did the modelling.
'''

teta = np.loadtxt(open("teta.csv","rb"), delimiter=",")
pr   = np.loadtxt(open("infiltration.csv","rb"), delimiter=",")*1e-2
cr   = np.loadtxt(open("upwelling.csv","rb"), delimiter=",")*1e-2
uptk = np.loadtxt(open("uptake.csv","rb"), delimiter=",")*1e-2
irr  = np.loadtxt(open("irrigations.csv","rb"), delimiter=",")
B    = np.loadtxt(open("cropbiomass.csv","rb"), delimiter=",")

#---------------------------------Model parameters----------------------------

# Depth of transport domain (from gwl up to soil surface), num & thickness of
# soil layers
gwl =  1.   
N   =  20    
dz  =  gwl/N  
z   =  np.arange(0., gwl + gwl/N, gwl/N)

# Simulation time (start, final &  # time step) 
ts = 0
dt = 1
t_dy = 365. 
t_final = float(raw_input("enter simulation time in days: "))
                                               

# Radioactive decay constants [1/dy]
TC_dcy = {'Cl': 0.00084,'Se':1.066e-06, 'I':4.33e-08}

# physical and chemical properties of the simulated soil column
e = 0.5             # soil porosity [cm3/cm3]
rho = 2650        # soil solid material density [kg/m3]
Kd = {'Cl': 1e-5,'Se':0.2, 'I':0.01}  # [m3/kg]
dl = 0.1            # diffusivity [m]
Dm = 1e-5          # hydrodynamic diffusion coeff [m2/d]

# activitiy concentration [Bq/cm3] in groundwater assumed constant during 
# simulation period
Cgw = 1             
switch = {'on': 1, 'off':0}

# crop specific parameters
# Mass interception factor for forage vegetation (dry weight) & food crops 
# (wet weight) from Table VII in SRS No. 19 (IAEA, 2001) 
f_intrcp = {'pasture': 3.0e4, 'foodcrops': 3.0e3} # [cm2/kg]
dy_swng = 236 # day of winter wheat sowing
dy_hrvs = 188 # day of crop harvest

# estimated residual (after harvest) biomass for wheat [kg/cm2]
Bmin = 0.1

#---------------------------------Model initialisation------------------------

T = []
A = []
S = np.zeros(N+1)

A0 = (N+1)*[0]  
T.append(ts)
A.append(A0)

# A = odeint(rhs, A0, T, full_output = True)

backend = "dopri5"

solver = ode(rhs)
solver.set_integrator(backend)  # nsteps=1
solver.set_initial_value(A0, ts)

while solver.successful() and solver.t < t_final:
  solver.integrate(solver.t + dt, step=1)
  T.append(solver.t)
  A.append(solver.y)
  
T = np.array(T)
A = np.array(A)

print time.time() - start_time, "seconds"  

'''
#---------------------------------Calculate ODE coefficients------------------

backend = 'dopri5'
sol = integrate.ode(rhs).set_integrator(backend) # moisture cont.
sol.set_initial_value(A0, ts) 

#---------------------------------Start solving ODE system--------------------

while sol.t < t_yr:
  
  # Solve dynamic system
  sol.integrate(sol.t + dt)
  
  np.append(T, sol.t)
  np.append(A, sol.y)
    
  # proceed to next year
  print 't = %g' %sol.t
  
#---------------------------------plotting------------------------------------

#fig = plt.figure()
#
#ax1 = plt.subplot(211)
#ax1.plot(T[:5000], A[:5000,-3], 'g-', label='top layer')
#ax1.plot(T[:5000], A[:5000,-2], 'r-', label='lower layer')
#plt.xlabel('time [day]')
#plt.ylabel('Activity [Bq]')
#plt.grid(True, 'major', 'both')
#
#ax2 = plt.subplot(212)
#ax2.plot(T[:5000], A[:5000,-1], 'g-', label='top layer')
#plt.xlabel('time [day]')
#plt.ylabel('Activity in crop [Bq]')
#plt.grid(True, 'major', 'both')
#
##locs = np.arange(z[0],z[-1]+10,10)
##plt.yticks(locs)
##plt.ylim(z[1], z[-1])
##plt.gca().invert_yaxis()

print time.time() - start_time, "seconds"       
#plt.show()
#

'''