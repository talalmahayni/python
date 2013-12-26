# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:28:50 2013
This code simulates solute (soil water containing a radionuclide) transport is
a soil column. The advection-dispersion model was adopted from Ireson & Butler
(2008): The effect of sorption on radionuclide uptake for different soil textu
-ers. 
@author: tamahain
"""
import time
import numpy as np
from scipy.linalg import solve
import matplotlib.pylab as plt

# start timing execusion
start_time = time.time()

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
      uptk [cm/cm/d]: uptake rate by plants. 
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
teta = np.loadtxt(open("teta_avg.csv","rb"), delimiter=",")
flux = np.loadtxt(open("flux.csv","rb"), delimiter=",")*1e-2
uptk = np.loadtxt(open("uptake.csv","rb"), delimiter=",")*1e-2
irr  = np.loadtxt(open("irrigations.csv","rb"), delimiter=",")
B    = np.loadtxt(open("cropbiomass.csv","rb"), delimiter=",")

#---------------------------------Model parameters----------------------------

# Depth of transport domain (from gwl up to soil surface), num & thickness of
# soil layers in [m]
gwl =  1.
N   =  20    
dz  =  gwl/N  
z   =  np.arange(0., gwl + gwl/N, gwl/N)

# Simulation time (start, final &  # time step) 
ts = 0
t_dy = 365. 
t_yr = float(raw_input("enter simulation time in years: "))
dt = 1                                             


# Radioactive decay constants [1/dy]
dcy = {'Cl': 0.00084,'Se':1.066e-06, 'I':4.33e-08}

# physical and chemical properties of the simulated soil column
e = 0.5                             # soil porosity [cm3/cm3]
rho = 2650                          # soil solid material density [kg/m3]
Kd = {'Cl': 0,'Se':0.2, 'I':0.01}   # [m3/kg]
dl = 0.1                            # longitudinal dispersivity [m]
Dm = 1e-5                           # molecular diffusion coefficient [m2/d]

# activitiy concentration [Bq/m3] in groundwater assumed constant during 
# simulation period
Cgw = 1             
switch = {'on': 1, 'off':0}

# crop specific parameters
# Mass interception factor for forage vegetation (dry weight) & food crops 
# (wet weight) from Table VII in SRS No. 19 (IAEA, 2001) 
f_intrcp = {'pasture': 3.0, 'foodcrops': 0.3}   # [m2/kg]
dy_swng = 236                                   # day of winter wheat sowing
dy_hrvs = 188                                   # day of crop harvest

# estimated residual (after harvest) biomass for wheat [kg/m2]
Bmin = 0.1

# PAHTWAY model formulat for the harvest fraction of crop biomass
# it is dy_hrvs -1 instead of dy_hrvs because of python 0-based indexing
f_hrvs = lambda t: 1-Bmin/B[t-1] if t == dy_hrvs else 0  

#---------------------------------Model initialisation------------------------

T  = np.array([ts])
A0 = np.zeros(N+1)  
A  = np.array([A0])  # A[-1] is activity in crop compartment [Bq]
S = np.zeros(N+1)
A_balance = np.array([0])

yr = 0

#---------------------------------Calculate ODE coefficients------------------

''' Fractional transfer rates are calculated following the approach of
    Klos (2008): GEMA3D-Landscape modelling for dose assessments. TCs
    represent the fraction of the inventory in one compartment that is
    transferred either to other compartments or out from the biosphere. 
    leach and upwel quantify solute movement in both directions by 
    advection and dispersion processes
'''
# Retardiation factor [-] 
R = 1 + (1-e)*rho*Kd['Se']/teta

# Darcy velocity [m/d] (excluding infiltration)
Fw = -flux[:,1:]

# Effective (retarded) average water velocity
Fr = Fw/R

# Hydrodynamic dispersion coefficient [m2/d]
D = dl*abs(Fw)/teta + Dm

# Corrected (adjusted for numerical dispersion effect) hydrodynamic dispersion
D = D - abs(Fw)*dz/2.

# Retarded hydrodynamic dispersion coefficient [m2/d]
Dr = D/R 

# Rate constants for passive uptake by crop (excluding surface node)   
r_uptk = switch['on']*uptk[:,1:]

# for mass balance calculations
sum_inflx = 0
sum_leach = 0
sum_uptk = 0 
sum_decy = 0

#---------------------------------Start solving ODE system--------------------
while yr < t_yr:
  
  t = ts # update dys counter

  while t < t_dy:
    
    # Elements of coefficient matrix: 
    # Advective transport from one segment to the next is determined from an 
    # estimate of the activity at the interface between the two segments using
    # central differencing (A[i+1]+A[i])/2
    
    # lower diagonal
    Ml  =  -(Dr[t,:-1]/(teta[t,:-1]*dz**2) + 0.5*Fr[t,:-1]/(teta[t,:-1]*dz))*dt
    
    # diagonal
    Mm  =  1 + (2*Dr[t,1:]/(teta[t,1:]*dz**2) + 0.5*(Fr[t,1:] - Fr[t,:-1])/(teta[t,1:]*dz) + r_uptk[t,1:]/teta[t,1:] + dcy['Se'])*dt
    
    # Handle top boundary:
    # 1st elements in the diagonals (365 diagonal rows: one per day) 
    Mm_tmp = 1 + (Dr[t,0]/(teta[t,0]*dz**2) + 0.5*Fr[t,0]/(teta[t,0]*dz) + r_uptk[t,0]/teta[t,0] + dcy['Se'])*dt
    
    # add 1st elements to the diagonal
    Mm = np.insert(Mm,0,Mm_tmp)

    # upper diagonal
    Mr  =  -(Dr[t,:-1]/(teta[t,:-1]*dz**2) - 0.5*Fr[t,:-1]/(teta[t,:-1]*dz))*dt
    
    # construct the unit response matrix 
    
    Ml = np.diagflat(Ml,-1)
    Mm = np.diagflat(Mm, 0)
    Mr = np.diagflat(Mr, 1)
     
    M = Ml + Mm + Mr
    
    # last row in unit response matrix for uptake rates (uptake removed from
    # every soil segment)
    M = np.vstack((M, -r_uptk[t,:]*dt))
    
    # add extra column to the right contains coefficients of As in the crop
    # activity balance equation. last ele of this row should be (1/dt+f): f is
    # fraction of crop biomass removed by harvesting. f should be made time
    # dependent function, a switch with value of 1 on day of harvest, and 0
    # otherwise    
    M_lst = np.append(np.zeros((M.shape[0]-1, 1)), 1)
    
    # to use the hstack, reshape the 1d array into 2d array
    M_lst = M_lst.reshape(M.shape[0], 1)
    
    # Stack ele of the TC_crop 2d array to evey row of TC array    
    M = np.hstack((M, M_lst))
    
    # Sourc/sink vector:
      
    # top boundary
    S[0]    = A[-1,0]  - switch['off']*Cgw*irr[t]*dt/dz 
    
    # middle segments
    S[1:-2] = A[-1,1:-2]  
    
    # bottom boundary
    S[-2]   = A[-1,-2] + (Dr[t,-1]/(teta[t,-1]*dz**2) - 0.5*Fr[t,-1]/(teta[t,-1]*dz))*dt*Cgw
    
    # Removal of crop by harvesting (treated explicitly)
    S[-1]   = A[-1,-1]*(1 - (f_hrvs(t) - dcy['Se'])*dt)      
            
    # Solve dynamic system
    
    # if the system of linear algebraic equations is not homogeneous
    A_tmp = solve(M, S)
    
    # don't reference A using t. this will empty the compartments at the begin
    # ning of every year since t becomes zero. use -1 index instead!
    
#    AB = inflx - (sum(A_tmp)-sum(A[-1,:]) + leach + uptak + decay)
    
    
    # Store results
    t += dt
    T = np.hstack([T, yr*t_dy+t])
    A = np.vstack([A, A_tmp])
    
   
    
  # proceed to next year
  yr += 1
  print 'yr = %g' %yr

# plotting
fig = plt.figure()

ax1 = plt.subplot(211)
ax1.plot(T/t_dy, A[:,0], 'b-', label='lower layer')
ax1.plot(T/t_dy, A[:,-3], 'g-', label='top layer')
ax1.plot(T/t_dy, A[:,-2], 'r-', label='lower layer')
plt.xlabel('time [year]')
plt.ylabel('Activity [Bq]')
plt.grid(True, 'major', 'both')

ax2 = plt.subplot(212)
ax2.plot(T/t_dy, A[:,-1], 'g-', label='top layer')
plt.xlabel('time [year]')
plt.ylabel('Activity in crop [Bq]')
plt.grid(True, 'major', 'both')

#locs = np.arange(z[0],z[-1]+10,10)
#plt.yticks(locs)
#plt.ylim(z[1], z[-1])
#plt.gca().invert_yaxis()

print time.time() - start_time, "seconds"       
plt.show()


