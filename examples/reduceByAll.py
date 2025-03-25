#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np

import functionsForExamples as fe
sys.path.append('../data-reduction-methods')
import workWithData as wd

path = fe.initialize()

Dataset = wd.TraversingData(**fe.preprocessSPLEENdata(path, 0.90, 120e3, 0.5))
reduced = Dataset.reduceByAll()

plt.plot([reduced[method]['p'] for method in reduced.keys()], 'o')
plt.grid()
plt.ylabel(r'$p$ [Pa]')
plt.xticks(np.arange(len(reduced.keys())), labels = [reduced[method]['method_abbr'] for method in reduced.keys()])
plt.show()
