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

from_p_p0_alpha_T0 = {
  'M'    : lambda values, fluid : ma_is(values['p'], values['p0'], fluid['gamma']),
  'T'    : lambda values, fluid : t(ma_is(values['p'], values['p0'], fluid['gamma']), values['T0'], fluid['gamma']),
  'rho'  : lambda values, fluid : rho(ma_is(values['p'], values['p0'], fluid['gamma']), values['p0']/values['T0']/fluid['r'], fluid['gamma']),
  'v_mag': lambda values, fluid : ma_is(values['p'], values['p0'], fluid['gamma']) * np.sqrt( fluid['gamma'] * fluid['r'] * t(ma_is(values['p'], values['p0'], fluid['gamma']), values['T0'], fluid['gamma'])), # np.sqrt(2*fluid['gamma']*fluid['r']*variables['T0']/(fluid['gamma']-1)*(1-(p/p0)**((fluid['gamma']-1)/fluid['gamma']))),  #
  'v_x'  : lambda values, fluid : ma_is(values['p'], values['p0'], fluid['gamma']) * np.sqrt( fluid['gamma'] * fluid['r'] * t(ma_is(values['p'], values['p0'], fluid['gamma']), values['T0'], fluid['gamma'])) * np.cos(values['alpha']),
  'v_y'  : lambda values, fluid : ma_is(values['p'], values['p0'], fluid['gamma']) * np.sqrt( fluid['gamma'] * fluid['r'] * t(ma_is(values['p'], values['p0'], fluid['gamma']), values['T0'], fluid['gamma'])) * np.sin(values['alpha'])
}

exponLosses = lambda gamma: (gamma-1)/gamma
losses = {
  'kineticEnergyLossCoefficient' :        lambda p_out, p_in, p0_out, p0_in, gamma : 1- (1-(p_out/p0_out)**exponLosses(gamma))/(1-(p_out/p0_in)**exponLosses(gamma)),
  'totalPressureLossCoefficient_dynIn' :  lambda p_out, p_in, p0_out, p0_in, gamma : (p0_in-p0_out)/(p0_in-p_in),
  'totalPressureLossCoefficient_dynOut' : lambda p_out, p_in, p0_out, p0_in, gamma : (p0_in-p0_out)/(p0_out-p_out),
  'totalPressureLossCoefficient_totIn' :  lambda p_out, p_in, p0_out, p0_in, gamma : (p0_in-p0_out)/(p0_in)
}

def kineticEnergyLossCoefficient(p_out, p_in, p0_out, p0_in, gamma=1.4):
  expon = (gamma-1)/gamma
  return 1- (1-(p_out/p0_out)**expon)/(1-(p_out/p0_in)**expon)

def totalPressureLossCoefficient_dynIn(p_out, p_in, p0_out, p0_in, gamma=1.4):
  return (p0_in-p0_out)/(p0_in-p_in)

def totalPressureLossCoefficient_dynOut(p_out, p_in, p0_out, p0_in, gamma=1.4):
  return (p0_in-p0_out)/(p0_out-p_out)

def totalPressureLossCoefficient_totIn(p_out, p_in, p0_out, p0_in, gamma=1.4):
  return (p0_in-p0_out)/(p0_in)

fluxesIntegrands = {
  'I_M' : lambda values, fluid : from_p_p0_alpha_T0['v_x'](values, fluid)    * from_p_p0_alpha_T0['rho'](values, fluid),
  'I_A' : lambda values, fluid : from_p_p0_alpha_T0['v_x'](values, fluid)**2 * from_p_p0_alpha_T0['rho'](values, fluid),
  'I_F' : lambda values, fluid : from_p_p0_alpha_T0['v_x'](values, fluid)**2 * from_p_p0_alpha_T0['rho'](values, fluid) + values['p'],
  'I_C' : lambda values, fluid : from_p_p0_alpha_T0['v_x'](values, fluid)    * from_p_p0_alpha_T0['rho'](values, fluid) * from_p_p0_alpha_T0['v_y'](values, fluid) ,
  'I_H' : lambda values, fluid : from_p_p0_alpha_T0['v_x'](values, fluid)    * from_p_p0_alpha_T0['rho'](values, fluid) * (values['p']/values['p0'])**((fluid['gamma']-1)/fluid['gamma']),
  'I_S' : lambda values, fluid : -fluid['r']*(from_p_p0_alpha_T0['v_x'](values, fluid)*np.log(values['p0']/values['p01'])           ),
}

normalize = lambda y: (y-y.min())/(y.max()-y.min())

if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')