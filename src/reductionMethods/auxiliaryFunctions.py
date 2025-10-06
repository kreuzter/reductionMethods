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
  'M'    : lambda p, p0, alpha, T0, fluid : ma_is(p, p0, fluid['gamma']),
  'T'    : lambda p, p0, alpha, T0, fluid : t(ma_is(p, p0, fluid['gamma']), T0, fluid['gamma']),
  'rho'  : lambda p, p0, alpha, T0, fluid : rho(ma_is(p, p0, fluid['gamma']), p0/T0/fluid['r'], fluid['gamma']),
  'v_mag': lambda p, p0, alpha, T0, fluid : ma_is(p, p0, fluid['gamma']) * np.sqrt( fluid['gamma'] * fluid['r'] * t(ma_is(p, p0, fluid['gamma']), T0, fluid['gamma'])), # np.sqrt(2*fluid['gamma']*fluid['r']*variables['T0']/(fluid['gamma']-1)*(1-(p/p0)**((fluid['gamma']-1)/fluid['gamma']))),  #
  'v_x'  : lambda p, p0, alpha, T0, fluid : ma_is(p, p0, fluid['gamma']) * np.sqrt( fluid['gamma'] * fluid['r'] * t(ma_is(p, p0, fluid['gamma']), T0, fluid['gamma'])) * np.cos(alpha),
  'v_y'  : lambda p, p0, alpha, T0, fluid : ma_is(p, p0, fluid['gamma']) * np.sqrt( fluid['gamma'] * fluid['r'] * t(ma_is(p, p0, fluid['gamma']), T0, fluid['gamma'])) * np.sin(alpha)
}



def losses(p_out, p_in, p0_out, p0_in, gamma=1.4):
  return [loss(p_out, p_in, p0_out, p0_in, gamma) for loss in [kineticEnergyLossCoefficient, 
                                                               totalPressureLossCoefficient_dynIn, 
                                                               totalPressureLossCoefficient_dynOut, 
                                                               totalPressureLossCoefficient_totIn]]

def kineticEnergyLossCoefficient(p_out, p_in, p0_out, p0_in, gamma=1.4):
  expon = (gamma-1)/gamma
  return 1- (1-(p_out/p0_out)**expon)/(1-(p_out/p0_in)**expon)

def totalPressureLossCoefficient_dynIn(p_out, p_in, p0_out, p0_in, gamma=1.4):
  return (p0_in-p0_out)/(p0_in-p_in)

def totalPressureLossCoefficient_dynOut(p_out, p_in, p0_out, p0_in, gamma=1.4):
  return (p0_in-p0_out)/(p0_out-p_out)

def totalPressureLossCoefficient_totIn(p_out, p_in, p0_out, p0_in, gamma=1.4):
  return (p0_in-p0_out)/(p0_in)

flux_mass     = lambda localMflux               :    (localMflux                          )
flux_momXP    = lambda localMflux, vX,  p       :    (localMflux*vX+p                     )
flux_momX     = lambda localMflux, vX           :    (localMflux*vX                       )
flux_momY     = lambda localMflux, vY           :    (localMflux*vY                       )
flux_enthalpy = lambda localMflux,  p, p0, gamma:    (localMflux*(p/p0)**((gamma-1)/gamma))
flux_enthalpyT= lambda localMflux,  T, cp       :    (localMflux*T*cp)
flux_entropy  = lambda localMflux, p0, p01, r   : -r*(localMflux*np.log(p0/p01)           )

#flux_gradientEntropy = lambda localMflux, p0, y, r : localMflux* 1/p0 *np.abs(np.gradient(p0, normalize(y)))

flux_gradientEntropy = lambda rho, w_x, p0, p01, y, r : w_x* np.abs(np.gradient(rho*np.log(p0/p01), normalize(y)))

fluxesIntegrands = {
  'I_M' : lambda p, p0, alpha, T0, p01, fluid : from_p_p0_alpha_T0['v_x'](p, p0, alpha, T0, fluid)    * from_p_p0_alpha_T0['rho'](p, p0, alpha, T0, fluid),
  'I_A' : lambda p, p0, alpha, T0, p01, fluid : from_p_p0_alpha_T0['v_x'](p, p0, alpha, T0, fluid)**2 * from_p_p0_alpha_T0['rho'](p, p0, alpha, T0, fluid),
  'I_F' : lambda p, p0, alpha, T0, p01, fluid : from_p_p0_alpha_T0['v_x'](p, p0, alpha, T0, fluid)**2 * from_p_p0_alpha_T0['rho'](p, p0, alpha, T0, fluid) + p,
  'I_C' : lambda p, p0, alpha, T0, p01, fluid : from_p_p0_alpha_T0['v_x'](p, p0, alpha, T0, fluid)    * from_p_p0_alpha_T0['rho'](p, p0, alpha, T0, fluid) * from_p_p0_alpha_T0['v_y'](p, p0, alpha, T0, fluid) ,
  'I_H' : lambda p, p0, alpha, T0, p01, fluid : from_p_p0_alpha_T0['v_x'](p, p0, alpha, T0, fluid)    * from_p_p0_alpha_T0['rho'](p, p0, alpha, T0, fluid) * (p/p0)**((fluid['gamma']-1)/fluid['gamma']),
  'I_S' : lambda p, p0, alpha, T0, p01, fluid : -fluid['r']*(from_p_p0_alpha_T0['v_x'](p, p0, alpha, T0, fluid)*np.log(p0/p01)           ),
}


normalize = lambda y: (y-y.min())/(y.max()-y.min())

def fluxesInt(data, p01, t, fluid):
  f = dict()

  lf = data['v_x']*data['rho']

  f['I_M' ] = np.trapz(flux_mass     (lf                                         ),  data['x'])/t
  f['I_F' ] = np.trapz(flux_momXP    (lf, data['v_x'],  data['p']                ),  data['x'])/t
  f['I_A' ] = np.trapz(flux_momX     (lf, data['v_x']                            ),  data['x'])/t
  f['I_C' ] = np.trapz(flux_momY     (lf, data['v_y']                            ),  data['x'])/t
  f['I_H' ] = np.trapz(flux_enthalpy (lf, data['p'] , data['p0'] , fluid['gamma']),  data['x'])/t
  f['I_HT'] = np.trapz(flux_enthalpyT(lf, data['T'] , fluid['cp']                ),  data['x'])/t
  f['I_S' ] = np.trapz(flux_entropy  (lf, data['p0'], p01        , fluid['r']    ),  data['x'])/t

  return f

def fluxesDisc(values, p01, fluid):
  f = dict()

  lf = values['v_x']*values['rho']

  f['I_M' ] = flux_mass     (lf)
  f['I_F' ] = flux_momXP    (lf, values['v_x'],  values['p'] )
  f['I_A' ] = flux_momX     (lf, values['v_x']      )
  f['I_C' ] = flux_momY     (lf, values['v_y']      )
  f['I_H' ] = flux_enthalpy (lf, values['p'] , values['p0'] , fluid['gamma'])
  f['I_HT'] = flux_enthalpyT(lf, values['T'] ,                fluid['cp']   )
  f['I_S' ] = flux_entropy  (lf, values['p0'], p01,           fluid['r']    )

  return f

if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')