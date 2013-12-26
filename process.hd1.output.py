# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:05:56 2013
script to process hydrus code outputs
extracts tetas, fluxes and sink
separate flux into two components: percolation and capillary  
@author: tamahain
"""

# import modules 
import numpy as np
# load input file into an object for processing, csv file opened in read mode
# elements of obj are floats
irr = raw_input('apply irrigation? [y/n]: ')
if irr == 'n':
  obj = np.loadtxt(open("raw_data.csv","rb"), delimiter=",")
else:
  obj = np.loadtxt(open("raw_data_irr.csv","rb"), delimiter=",")
# number of days in the output file (tf-ts)+1.
numdy = int(raw_input('enter number of days: '))
numdy += 1 #counting start from zero 
# number of nodes in the input & output files
numnod = int(raw_input('enter number of nodes: '))

# number of soil layers is 1 less than numnod
numlay = numnod-1
# input file contains 4 columns nod, teta, flux and sink (uptake)
numcols = 4
# initialise containers (lists) for outputs 
teta_dy = []
flux_dy = []
sink_dy = []
# numpy arrays for final outputs: better for later use
teta_dy_avg = np.zeros((numdy, numlay))
pr_dy = np.zeros((numdy, numnod))
cr_dy = np.zeros((numdy, numnod))
# first loop over days then in each day over every nod, total num of rows
# in the input file is numdy*numnode
for r in xrange(numdy):  
  # initialise temporary storage tanks
  teta_tmp = []
  flux_tmp = []
  sink_tmp = []  
  for nod in xrange(numnod):    
    teta_tmp += [obj[r*numnod+nod][1]]
    flux_tmp += [obj[r*numnod+nod][2]]
    sink_tmp += [obj[r*numnod+nod][3]]    
  teta_dy.append(teta_tmp)
  flux_dy.append(flux_tmp)
  sink_dy.append(sink_tmp)  
# convert lists to arrays
teta_dy = np.array(teta_dy)
flux_dy = -np.array(flux_dy) #minus sign to counteract hydrus sign convention
sink_dy = np.array(sink_dy)
# from nodal tetas calculate teta for layers
for dy in xrange(numdy):
  teta_dy_avg[dy, :] = 0.5*(teta_dy[dy,1:]+teta_dy[dy,:-1])    
  # separate flux into percolation and capillary flux
  for nod in xrange(numnod):
    if flux_dy[dy,nod] > 0:
      pr_dy[dy, nod] = flux_dy[dy,nod] 
      cr_dy[dy, nod] = 0
    else:
      pr_dy[dy, nod] = 0
      cr_dy[dy, nod] = -flux_dy[dy,nod]
# write outputs/processed data
if irr == 'n':
  np.savetxt('teta.csv', teta_dy, delimiter=',')
  np.savetxt('flux.csv', flux_dy, delimiter=',')
  np.savetxt('uptake.csv', sink_dy, delimiter=',')
  
else:
  np.savetxt('teta_irr.csv', teta_dy, delimiter=',')
  np.savetxt('flux_irr.csv', flux_dy, delimiter=',')
  np.savetxt('uptake_irr.csv', sink_dy, delimiter=',')
  
'''
think how to give headers and insert a column at the most left of the csv 
sheet for day of year 1,365
'''