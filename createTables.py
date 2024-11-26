import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import auxiliaryFunctions as aux
from workWithData import TraversingData

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
  "font.serif" : "Computer Modern",
  "font.size"  : 12,
  'axes.spines.right' : False,
  'axes.spines.top'   : False,
  'axes.spines.left'  : False,
  'axes.spines.bottom': False,
})

schreiber84 = 'data/schreiber84/p0-1.0e5_Ma-0.95_B-147.0.csv'
data = TraversingData(pd.read_csv(schreiber84, delimiter=',', header=0), turbine=False,
               add={'T0':303.15, 'pitch':0.031480, 'p01':1e5, 'p1':aux.p(0.95)*1e5})
print(data.rawData)
print(data.reduction_vzlu())
print(data.reduction_massFlux())

fig, ax = data.plot()

plt.show()