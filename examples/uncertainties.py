#!/usr/bin/env python3
import sys
import numpy as np

import matplotlib.pyplot as plt
from reductionMethods import workWithData as wd
from reductionMethods import auxiliaryFunctions as aux

data = np.loadtxt('data/kobra_a1-61.0_p1-63671_p01-97672.csv', delimiter=';', skiprows=1)
t0 = data[0,-1]/aux.t(aux.ma_is(data[0,1], data[0,0]))

Dataset = wd.TraversingData( { 'p' : data[:,1], 'p0' :data[:,0], 'alpha':np.deg2rad(data[:,2]), 'x':np.linspace(0,1,len(data[:,0]))}, 
                            inlet={'p':63671, 'p0':97672, 'T0':t0}, 
                            uncertainties={'p':50, 'p0':50, 'alpha':np.deg2rad(0.6), 'p1':50, 'p01':17, 'T0':0.3})

var2check = 'T'
print(Dataset.rawData[var2check][0])
print(Dataset.data_uncertainties[var2check][0]*2)
Dataset.fluxes()
print(Dataset.trueFluxesUncertainties)