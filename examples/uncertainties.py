#!/usr/bin/env python3
import sys
import numpy as np

from pprint import pprint

import matplotlib.pyplot as plt
from reductionMethods import workWithData as wd
from reductionMethods import auxiliaryFunctions as aux

data = np.loadtxt('/media/kreuzter/volume/PhD/reductionMethods/examples/data/trw2_a1-1.1_p1-92210_p01-96061.csv', delimiter=';', skiprows=1)
t0 = data[0,-1]/aux.t(aux.ma_is(data[0,1], data[0,0]))

Dataset = wd.TraversingData( { 'p' : data[:,1], 'p0' :data[:,0], 'alpha':np.deg2rad(data[:,2]), 'x':np.linspace(0,1,len(data[:,0]))}, 
                            inlet={'p':92099., 'p0':96091., 'T0':t0}, 
                            uncertainties={'alpha':np.deg2rad(0.6)} 
                            )

var2check = 'T'
print(f'T[0]= {Dataset.rawData[var2check][0] :.3f} +/- {Dataset.rawDataUncertainties[var2check][0]*2 :.3f} K')

Dataset.integralFluxes()
for flux in Dataset.trueFluxesUncertainties.keys():
  print(f'integrated {flux} = {Dataset.trueFluxes[flux] :.3f} +/- {Dataset.trueFluxesUncertainties[flux] :.3f}, uncertainty is {Dataset.trueFluxesUncertainties[flux]/Dataset.trueFluxes[flux]*100 :.3f} %')
reses = Dataset.reduceByAll()

def printResult(res):
  
  print()
  print(res['method_name'])
  #pprint(res)
  for v in ['p0', 'p', 'rho', 'T', 'v_mag', 'v_x', 'v_y', 'M']:
    print(f'   <{v}> = {res[v] :.6f} +/- {res["uncertainties"][v] :.6f}, uncertainty is {res["uncertainties"][v]/res[v] *100 :.3f} %')
  print(f'   <alpha> = {np.rad2deg(res['alpha']) :.6f} +/- {np.rad2deg(res["uncertainties"]["alpha"]) :.6f}, uncertainty is {res["uncertainties"]['alpha']/res['alpha'] *100 :.3f} %')  
  #pprint(res['fluxes'])

for res in reses.keys():
  printResult(reses[res])
