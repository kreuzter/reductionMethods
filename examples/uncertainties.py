#!/usr/bin/env python3
import sys
import numpy as np

import json

from pprint import pprint

import matplotlib.pyplot as plt
from reductionMethods import workWithData as wd
from reductionMethods import auxiliaryFunctions as aux

fileNames = ['kobra_a1-57.7_p1-64462_p01-98428','kobra_a1-59.0_p1-64615_p01-97992','kobra_a1-61.0_p1-63671_p01-97672','kobra_a1-62.0_p1-63435_p01-97388','kobra_a1-63.0_p1-64341_p01-97789','trw2_a1-1.1_p1-92210_p01-96061','trw2_a1-1.301_p1-92149_p01-96062','trw2_a1-1.495_p1-92073_p01-96066','trw2_a1-1.709_p1-92099_p01-96091']

def printResult(res):
  
  print()
  print(res['method_name'])
  for v in ['p0', 'p', 'rho', 'T', 'v_mag', 'v_x', 'v_y', 'M', 'kineticEnergyLossCoefficient','totalPressureLossCoefficient_dynIn' ,'totalPressureLossCoefficient_dynOut','totalPressureLossCoefficient_totIn', 'I_M','I_A','I_F','I_C','I_H','I_S' ]:
    print(f'   <{v}> = {res[v] :.6f} +/- {res["uncertainties"][v] :.6f}, uncertainty is {res["uncertainties"][v]/res[v] *100 :.3f} %')
  print(f'   <alpha> = {np.rad2deg(res['alpha']) :.6f} +/- {np.rad2deg(res["uncertainties"]["alpha"]) :.6f}')  

for i,fileName in enumerate(fileNames):

  inlet = { it.split('-')[0] : float(it.split('-')[1]) for it in fileName.split('_')[1:]}
  for p in ['p', 'p0']:
    inlet[p] = inlet.pop(f'{p}1')

  data = np.loadtxt(f'data/{fileName}.csv', delimiter=';', skiprows=1)
  t0 = data[0,-1]/aux.t(aux.ma_is(data[0,1], data[0,0]))
  inlet['T0']

  Dataset = wd.TraversingData( { 'p' : data[:,1], 'p0' :data[:,0], 'alpha':np.deg2rad(data[:,2]), 'x':np.linspace(0,1,len(data[:,0]))}, 
                              inlet=inlet, 
                              uncertainties={'alpha':np.deg2rad(0.6), 'p':50., 'p0':50., 'p01':17., 'T0':0.3} 
                              )

  var2check = 'T'
  print(f'T[0]= {Dataset.rawData[var2check][0] :.3f} +/- {Dataset.rawDataUncertainties[var2check][0]*2 :.3f} K')

  Dataset.integralFluxes()
  for flux in Dataset.trueFluxesUncertainties.keys():
    print(f'integrated {flux} = {Dataset.trueFluxes[flux] :.3f} +/- {Dataset.trueFluxesUncertainties[flux] :.3f}, uncertainty is {Dataset.trueFluxesUncertainties[flux]/Dataset.trueFluxes[flux]*100 :.3f} %')
  reses = Dataset.reduceByAll()

  with open(f'reductionMethods/data/jsons/{fileName}.json', 'w') as outfile:
    json.dump(reses, outfile)
