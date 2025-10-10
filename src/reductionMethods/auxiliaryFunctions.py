#! /usr/bin/env python3

import numpy as np

def ma_is(p, p0=1, gamma=1.4):
  return np.sqrt( 2/(gamma-1) * ((p0/p)**((gamma-1)/gamma) -1) )

gd_isen = lambda ma, gamma : 2 / ( ma**2*(gamma-1) +2 )

def t(ma, t0=1, gamma = 1.4):
  return t0 * gd_isen(ma, gamma) 

def p(ma, p0=1, gamma = 1.4):
  return p0 * gd_isen(ma, gamma) ** (gamma/(gamma-1))

def rho(ma, rho0=1, gamma = 1.4):
  return rho0 * gd_isen(ma, gamma) ** (1/(gamma-1))

s = lambda p0, p01, r : -r*np.log(p0/p01)
rho_id = lambda p, t, r : p/r/t

exponLosses = lambda gamma: (gamma-1)/gamma
losses = {
  'kineticEnergyLossCoefficient'        : lambda p_out, p_in, p0_out, p0_in, gamma : 1- (1-(p_out/p0_out)**exponLosses(gamma))/(1-(p_out/p0_in)**exponLosses(gamma)),
  'totalPressureLossCoefficient_dynIn'  : lambda p_out, p_in, p0_out, p0_in, gamma : (p0_in-p0_out)/(p0_in-p_in),
  'totalPressureLossCoefficient_dynOut' : lambda p_out, p_in, p0_out, p0_in, gamma : (p0_in-p0_out)/(p0_out-p_out),
  'totalPressureLossCoefficient_totIn'  : lambda p_out, p_in, p0_out, p0_in, gamma : (p0_in-p0_out)/(p0_in)
}

normalize = lambda y: (y-y.min())/(y.max()-y.min())

if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')