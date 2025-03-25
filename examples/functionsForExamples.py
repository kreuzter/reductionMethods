import numpy as np
import pandas as pd
import sys
import argparse

sys.path.append('../data-reduction-methods')
import auxiliaryFunctions as aux

fluid = lambda gamma = 1.4, r = 8314.3/28.96 : {'gamma':gamma, 'r':r, 'cp':gamma*r/(gamma-1)}

def initialize() -> str:
  parser = argparse.ArgumentParser()

  parser.add_argument('path', metavar='path', type=str, help='path to SPLEEN (SPLEEN_HighSpeedTurbineCascade_Database_v5) directory')
  args = parser.parse_args()
  
  return args.path

def preprocessSPLEENdata(basicPath:str, mach:float, reynolds:float, span:float) -> dict :
  air = fluid()

  regime=f'SPLEEN_C1_NC_St000_Re{reynolds/1e3:.0f}_M0{mach*1e2:.0f}_PL06_L5HP_s{span*1e4:04.0f}'
  path = basicPath+'Experimental_DataBase/1_SPLEENC1_NC_WGOFF/PL06/L5HP/Area/'+regime+'.xlsx'
  
  data = pd.read_excel(path).to_numpy()

  #manipulation with data to get a classical triple: p, p0, alpha 
  y     = data[:,1]
  alpha = np.deg2rad(data[:,3]+53.8)
  p2p0  = data[:,-2]/data[:,-3]
  ma    = aux.ma_is(p2p0)
  t     =   (data[:,6]/np.cos(alpha)/ma)**2/air['gamma']/air['r']
  a_t0  = t+(data[:,6]/np.cos(alpha))**2/2/air['cp']
  t0    = np.mean(a_t0)
  p     = data[:,5]*air['r']*t
  p0    = p/p2p0

  p01 = np.mean(p/data[:,-2])

  dataDict = {
    'x' :    y     ,
    'p' :    p     ,
    'p0':    p0    ,
    'alpha': alpha ,
  }

  inlet = {'p0':p01, 'T0':t0, 'p':aux.p(mach, p01)}

  return {'data':dataDict, 'inlet':inlet, 'fluid':air}


if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')