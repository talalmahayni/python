"""
Created on Mon Nov 25 13:28:50 2013
This code simulates solute (soil water containing a radionuclide) transport is
a soil column. The advection-dispersion model was adopted from Ireson & Butler
(2008): The effect of sorption on radionuclide uptake for different soil textu
-ers.
An implicit (backward Euler) forward finite difference scheme is used. This 
scheme is one order in time and space.
dc/dt = [(d**2/dz**2)*J_dis-d/dz(J_adv)-J_upt-c*dteta/dt]/(R*teta)-decay*c
J_dis = -D*dc/dz      (dispersive flux)
J_adv = v*c           (advective flux)
J_upt = k_upt*c       (uptake by plant flux)
@author: tamahain
"""

#---------------------------------import modules------------------------------
import sys
import time
import numpy as np
from scipy.linalg import solve
import matplotlib.pylab as plt

#start timing execusion
start_time = time.time()

""" 
hydrological data are outputs from Hydrus 1D code for Sutton-Bonington site in
the UK. A 100 cm sandy loam soil column with winter wheat crop was simulated
for a period of one year assuming water initially at field capacity +
irrigation. For the boundaries, site-specific weather data was used and static
water table. The Hydrus outputs used here are:
teta [m3/m3]: moisture content per node 
flux [m3/d]: nodal Darcy fluxes (+ downward and - upward)
uptake[m3/m3/d]: rate of water uptake by plant roots. x dz to convert to flux
irr[cm3/cm2/d]: irrigation rates are stored in a csv file and used to simulate 
spikes/inputs of activity into the soil folloiwng irrigation events 
B [kg/m2]: crop (wheat) total biomass simulated using Sirius crop model. 
Details of this simulation are unkown since Niel Crout did the modelling.
"""
#user inputs
irr = raw_input('apply irrigation? [y/n]: ') 
if irr != 'y' and irr !='n':
  print 'only y or n are valid'
  sys.exit()
  
s_base_flux = raw_input('allow bottom influx? [y/n]: ')
if s_base_flux != 'y' and s_base_flux !='n':
  print 'only y or n are valid'
  sys.exit()
  
ele = raw_input('Select element (Cl, Se, I): ')

t_yr = float(raw_input('Simulation time in years: '))

switch = {'on': 1, 'off':0}

#---------------------------------specifying hydrological fluxes--------------

if irr == 'n': 
  teta = np.loadtxt(open("teta.csv","rb"), delimiter=",")
  flux = np.loadtxt(open("flux.csv","rb"), delimiter=",")
  uptk = np.loadtxt(open("uptake.csv","rb"), delimiter=",")
  irr  = switch['off']*np.loadtxt(open("irrigations.csv","rb"), delimiter=",")
else:
  teta = np.loadtxt(open("teta_irr.csv","rb"), delimiter=",")
  flux = np.loadtxt(open("flux_irr.csv","rb"), delimiter=",")
  uptk = np.loadtxt(open("uptake_irr.csv","rb"), delimiter=",")
  irr  = switch['on']*np.loadtxt(open("irrigations.csv","rb"), delimiter=",")

B  = np.loadtxt(open('cropbiomass.csv','rb'), delimiter=",")
rd = np.loadtxt(open('rootdepth.csv','rb'), delimiter=",")


#---------------------------------Model parameters----------------------------

# Depth of transport domain (from gwl up to soil surface), num & thickness of
# soil layers in [m]

gwl =  1.
N   =  100    
dz  =  gwl/N  
z   =  np.arange(0., gwl + gwl/N, gwl/N)

# Simulation time (start, final &  # time step) 

ts = 1
t_dy = 365. + ts #we're starting from day 1 & there're 365 days/year
dt = 1                                             

# physical and chemical properties of the simulated soil column
e = 0.5                             #soil porosity [m3/m3]
rho = 2650                          #soil solid material density [kg/m3]
Kd = {'Cl': 0,'Se':0.2, 'I':0.01}
dcy = {'Cl': 84.0e-5,'Se':1.066e-06, 'I':4.33e-08} #decay const [1/dy]
dl = 0.1                            #longitudinal dispersivity [m] (tenth of
                                    #the flow domain)
Dm = 1e-5                           #molecular diffusion coefficient [m2/d] in  
                                    #porewter 

# activitiy concentration [Bq/m3] in groundwater assumed constant during 
# simulation period
Cgw = 1                       #radionuclide conc. in surface & ground waters

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

#---------------------------------Calculate ODE coefficients------------------

''' Fractional transfer rates are calculated following the approach of
    Klos (2008): GEMA3D-Landscape modelling for dose assessments. TCs
    represent the fraction of the inventory in one compartment that is
    transferred either to other compartments or out from the biosphere. 
    leach and upwel quantify solute movement in both directions by 
    advection and dispersion processes
'''
#nodal Darcy velocity [m/d] calculated in HYDRUS-1D (including infiltration)
#flow rate of water per unit area (hereafter referred to as the water flux)
fw = flux

#soil moisture @ cell centres (averaging teta @ interfaces)
#alternatively teta from HYDRUS output file could be used
teta_  = 0.5*(teta[:,1:]+teta[:,:-1])
dteta_ = teta_[1:,:]-teta_[:-1,:]      #change in moisture over a single day

#366 values are needed (ts=1). 
dteta_ = np.insert(dteta_,0,np.zeros(dteta_.shape[1]),axis=0)  

#Retardiation factor [-] 
R = 1 + (1-e)*rho*Kd[ele]/teta_

#dispersion coefficient [m2/d] is sum of soil molecular diffusion (accounts
#for tortuousity) and mechanical dispersion (teta @ cell interface)
D = dl*abs(fw) + teta*Dm

# Corrected (adjusted for numerical dispersion effect) hydrodynamic dispersion
D = D - abs(fw)*dz/2.

# Rate of passive uptake (via transpiration) by crop    
r_uptk = switch['on']*uptk[:,1:]  #excluding surface node

#---------------------------------Model initialisation------------------------

#simulation times
T  = np.array([ts])

#activitiy concentration in porewater [Bq/m3]
cpw0 = np.zeros(N)
cpw  = np.array([cpw0])  

#total activitiy in porewater [Bq/m3]
Apwt0 = 0.
Apwt  = np.array([Apwt0])  

#activity in crop compartment [Bq]
Ap0 = 0
Ap = np.array([Ap0])

#sources/sinks
S = np.zeros(N+1)

MB = 0

yr = 0

#---------------------------------Start solving ODE system--------------------
while yr < t_yr:
  
  t = ts # update dys counter

  while t < t_dy:
    
    #coefficients of the response matrix: 
      
    #subdiagonal: coefficients of c[i-1]   
    #diffusion
    Ml = -D[t,1:-1] / dz
    #advetion
    Ml -= np.maximum(0, fw[t,1:-1]) 
    #scaling operator (retardation, 2nd derivative dz, time step)
    Ml *= dt / (R[t,1:] * teta_[t,1:] * dz) 
    
    # main diagonal: coefficients of c[i]
    #diffusion
    Mm =  (D[t, 1:-2] + D[t, 2:-1]) / dz
    #advection
    Mm += (np.maximum(0, fw[t, 2:-1]) - np.minimum(0, fw[t, 1:-2]))
    #root uptake
    Mm += r_uptk[t, 1:-1] 
    #scaling operator (retardation, 2nd derivative dz, time step)
    Mm *= dt / (R[t, 1:-1] * teta_[t, 1:-1] * dz)
    #radiodecay of both liquid & sorbed phases
    Mm += dcy[ele] * dt    
    Mm += 1.
    
    #coefficients of boundary nodes:
    #top cell:
      
    Mm_top = D[t,1] / dz
    Mm_top += np.maximum(0, fw[t, 1])
    #control outflux through soil surface using a switch 
    Mm_top -= switch['off'] * np.minimum(0, fw[t, 0])
    Mm_top += r_uptk[t, 0]
    Mm_top *= dt / (R[t, 0] * teta_[t, 0] * dz)
    Mm_top += dcy[ele] * dt
    Mm_top += 1.
    
    #bottom cell:
    Mm_bot = D[t,-2] / dz
    Mm_bot += np.maximum(0, fw[t, -1]) - np.minimum(0, fw[t, -2])
    Mm_bot += r_uptk[t, -1]
    Mm_bot *= dt / (R[t, -1] * teta_[t, -1] * dz)
    Mm_bot += dcy[ele] * dt
    Mm_bot += 1.
    
    #update coefficient matrix
    Mm = np.insert(Mm, 0, Mm_top)
    Mm = np.append(Mm, Mm_bot)

    #super diagonal: coefficients of c[i+1]    
    Mr = -D[t, 1:-1] / dz
    Mr += np.minimum(0, fw[t, 1:-1])
    Mr *= dt / (R[t, 1:] * teta_[t, 1:] * dz)
    
    #assembel the unit response matrix from sub, super and diagonal coefficients     
    Ml = np.diagflat(Ml, -1)
    Mm = np.diagflat(Mm, 0)
    Mr = np.diagflat(Mr, 1) 
    
    M = Ml + Mm + Mr
    
    #last row in unit response matrix for uptake rates (uptake removed from
    #every soil segment)    
    M = np.vstack((M, -r_uptk[t, :] * dt * dz))
    
    #add extra column to the right contains coefficients of c in the crop
    #equation. last ele of this row should be (1/dt+f): f is fraction of crop
    #biomass removed by harvesting. f should be made time dependent function
    #a switch with value of 1 on day of harvest, and 0 otherwise    
    M_lst = np.append(np.zeros((M.shape[0]-1, 1)), 1)
    
    # to use the hstack, reshape the 1d array into 2d array    
    M_lst = M_lst.reshape(M.shape[0], 1)
    
    # Stack ele of the TC_crop 2d array to evey row of TC array  
    M = np.hstack((M, M_lst))
    
    # coefficients of RHS vector:      
    # top boundary
    #first inputs with infiltration of rainwater    
    S[0] = np.maximum(0, fw[t, 0]) * Cgw * dt * switch['off'] 
    S[0] *= dt / (R[t, 0] * teta_[t, 0] * dz)
    #then add inputs with irrigations    
    S[0] += irr[t] * Cgw * dt                                
    S[0] += cpw[-1,0] * (1 - dteta_[t, 0]/(R[t, 0] * teta_[t, 0] * dt))
    
    # middle segments    
    S[1:-2] = cpw[-1, 1:-1] * (1 - dteta_[t, 1:-1] / (R[t, 1:-1] * teta_[t, 1:-1] * dt))  
    
    # bottom boundary
    #following the approach of Ireson and Butler (2008): mixed boundary
    #2nd term is mass flux per unit area [Bq/m2/d]
    if s_base_flux == 'n':
      S[-2] = - np.minimum(0, fw[t,-1]) * Cgw * dt * switch['off']
    else:    
      S[-2] = - np.minimum(0, fw[t,-1]) * Cgw * dt * switch['on']
      
    S[-2] *= dt / (R[t,-1] * teta_[t, -1] * dz)    
    S[-2] += cpw[-1, -1] * (1 - dteta_[t, -1] / (R[t, -1] * teta_[t, -1] * dt))
    
    # Removal of crop by harvesting (treated explicitly)    
    S[-1]   = Ap[-1] * (1 - (f_hrvs(t) - dcy[ele]) * dt)      
            
    # Solve system of algebraic equations (system dynamics)    
    sol = solve(M, S)
    
    #allocate solutions 
    cpw_tmp = sol[:-1]                          #activitiy conc. in porewater [Bq/m3]
    Apwt_tmp = sum(cpw_tmp * teta_[t, :] * dz)  #total activitiy in porewater [Bq]
    Ap_tmp  = sol[-1]                           #total activitiy in crop [Bq]

    """    
    # don't reference A using t. this will empty the compartments at the begin
    # ning of every year since t becomes zero. use -1 index instead!
    influx = Cgw*fw[t,-1]
    leach  = c[t,-2]*np.maximum(0,fw[t,-1])
    uptak  = c[t,:-1]*r_uptk[t,:]/(R[t,:]*teta_[t,:])*dt
    decay  = c[t,:-1]*dcy['Se']
    #print (sum(c_tmp[:-1]*dz)-sum(c[-1,:-1]*dz) - influx + leach + sum(uptak) + sum(decay))
    
    """ 
    
    #now store results    
    t += dt
    T     = np.hstack([T, yr * t_dy + t])
    cpw   = np.vstack([cpw, cpw_tmp])
    Apwt  = np.vstack([Apwt, Apwt_tmp])
    Ap    = np.hstack([Ap, Ap_tmp])
    
       
  #proceed to next year
  yr += 1
  print 'yr = %g' %yr

#---------------------------------postprocessing------------------------------

#sorbed activity concentration [Bq/kg]
cs = Kd[ele] * cpw


#print execusion time
print time.time() - start_time, "seconds" 

# plotting
plt.figure('Concentration dynamics')
ax1 = plt.subplot(211)
ax1.plot(T/t_dy, Apwt)
plt.xlabel('Time [year]')
plt.ylabel('Total dissolved activitiy in soil column $[Bq]$')
plt.grid(True, 'major', 'both')
plt.title('Total dissolved activitiy in soil column')

ax2 = plt.subplot(212)
ax2.plot(T/t_dy, Ap, 'g-', label='Crop')
plt.xlabel('Time [year]')
plt.ylabel('Activity in crop $[Bq]$')
plt.grid(True, 'major', 'both')
plt.legend(loc='best')

plt.figure('Concentration profiles')
locs = np.arange(z[0], z[-1] + 10*dz, 10*dz)
plt.yticks(locs)
plt.ylim(z[1], z[-1])
plt.gca().invert_yaxis()
plt.xlabel('Sorbed activity concentration $[Bq \ kg^{-1}]$')
plt.ylabel('Depth  $[m]$')
plt.grid(True, 'major', 'both')
plt.plot(1e3*cs[0,:],z[1:],'b--', label=r'$t = 0$')
plt.plot(1e3*cs[len(cs[:])/2,:],z[1:],'c--', label=r'$t=t_{f}/2$')
plt.plot(1e3*cs[-1,:],z[1:],'r--', label=r'$t=t_{f}$')
plt.legend(loc='best')
plt.show()