#!/usr/bin/env python3
import sys
import numpy as np

from pprint import pprint

import matplotlib.pyplot as plt
from reductionMethods import workWithData as wd
from reductionMethods import auxiliaryFunctions as aux

data = np.loadtxt('data/kobra_a1-61.0_p1-63671_p01-97672.csv', delimiter=';', skiprows=1)
t0 = data[0,-1]/aux.t(aux.ma_is(data[0,1], data[0,0]))

Dataset = wd.TraversingData( { 'p' : data[:,1], 'p0' :data[:,0], 'alpha':np.deg2rad(data[:,2]), 'x':np.linspace(0,1,len(data[:,0]))}, 
                            inlet={'p':63671, 'p0':97672, 'T0':t0}, 
                            uncertainties={'p':50, 'p0':50, 'alpha':np.deg2rad(0.6), 'p1':50, 'p01':17, 'T0':0.3})

var2check = 'T'
print(f'T[0]= {Dataset.rawData[var2check][0] :.3f} +/- {Dataset.data_uncertainties[var2check][0]*2 :.3f} K')
Dataset.integralFluxes()
for flux in Dataset.trueFluxesUncertainties.keys():
  print(f'integrated {flux} = {Dataset.trueFluxes[flux] :.3f} +/- {Dataset.trueFluxesUncertainties[flux] :.3f}, uncertainty is {Dataset.trueFluxesUncertainties[flux]/Dataset.trueFluxes[flux]*100 :.3f} %')

for nameReduction in ['massFlux', 'area', 'momentum', 'enthalpy', 'entropy']:
  res = Dataset.reduction_universalAveraging(nameReduction)
  print()
  print(res['method_name'])
  #pprint(res)
  for v in ['p', 'p0', 'rho', 'T', 'v_mag', 'v_x', 'v_y', 'M']:
    print(f'   <{v}> = {res[v] :.3f} +/- {res["uncertainties"][v] :.3f}, uncertainty is {res["uncertainties"][v]/res[v] *100 :.3f} %')
  print()  
  for fluxName in Dataset.trueFluxes.keys():
    print(f'   {fluxName} = {res["fluxes"][fluxName] :.3f} +/- {res["fluxesUncertainties"][fluxName] :.3f}, uncertainty is {res["fluxesUncertainties"][fluxName]/res["fluxes"][fluxName] *100 :.3f} %')

res = Dataset.reduction_vzlu(None)
print()
print(res['method_name'])
#pprint(res)
for v in ['p', 'p0', 'rho', 'T', 'v_mag', 'v_x', 'v_y', 'M']:
  print(f'   <{v}> = {res[v] :.3f} +/- {res["uncertainties"][v] :.3f}, uncertainty is {res["uncertainties"][v]/res[v] *100 :.3f} %')
print()  
for fluxName in Dataset.trueFluxes.keys():
  print(f'   {fluxName} = {res["fluxes"][fluxName] :.3f} +/- {res["fluxesUncertainties"][fluxName] :.3f}, uncertainty is {res["fluxesUncertainties"][fluxName]/res["fluxes"][fluxName] *100 :.3f} %')


res = Dataset.reduction_momentumMethod(True)
print()
print(res['method_name'])
#pprint(res)
for v in ['p', 'p0', 'rho', 'T', 'v_mag', 'v_x', 'v_y', 'M']:
  print(f'   <{v}> = {res[v] :.3f} +/- {res["uncertainties"][v] :.3f}, uncertainty is {res["uncertainties"][v]/res[v] *100 :.3f} %')
print()  
for fluxName in Dataset.trueFluxes.keys():
  print(f'   {fluxName} = {res["fluxes"][fluxName] :.3f} +/- {res["fluxesUncertainties"][fluxName] :.3f}, uncertainty is {res["fluxesUncertainties"][fluxName]/res["fluxes"][fluxName] *100 :.3f} %')