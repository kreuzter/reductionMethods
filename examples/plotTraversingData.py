#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt

import functionsForExamples as fe
sys.path.append('../data-reduction-methods')
import workWithData as wd

path = fe.initialize()

Dataset = wd.TraversingData(**fe.preprocessSPLEENdata(path, 0.90, 120e3, 0.5))
fig, ax = Dataset.plotRawData([['p', 'p0'], ['v_x', 'v_y'], ['T']], ylabels=[None,r'$w$ [m/s]', None])
plt.show()

re = 70e3
for mach, span in zip([0.70, 0.70, 0.80, 0.90, 0.95], [1e-2, 0.5, 0.5, 0.5, 0.5]):
  
  Dataset = wd.TraversingData(**fe.preprocessSPLEENdata(path, mach, re, span))

  figi, axi = Dataset.plotRawData([['p'], ['p0'], ['alpha'], ['M']], 
                                  figsize=(4.13,5),
                                  figax=None if span == 1e-2 else (figi, axi), 
                                  legenPrepend=f'Re = {re}, M = {mach :.2f}, Z = {span:.2f} %', 
                                  cycLinestyle=['-', '--', ':', '-.', (0, (3, 10, 1, 10))],
                                  legendkwargs={'loc':'lower right', 'bbox_to_anchor':(1.05, 1),'ncols':1},
                                  rcParams_user={"font.size" : 10})
  del Dataset
  
plt.show()