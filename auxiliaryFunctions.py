import numpy as np
import pandas as pd

def ma_is(p, p0=1, gamma=1.4):
  return np.sqrt( 2/(gamma-1) * ((p0/p)**((gamma-1)/gamma) -1) )

def gd_isen(ma, gamma):
  return 2 / ( ma**2*(gamma-1) +2 )

def t(ma, t0=1, gamma = 1.4):
  return t0 * gd_isen(ma, gamma) 

def p(ma, p0=1, gamma = 1.4):
  return p0 * gd_isen(ma, gamma) ** (gamma/(gamma-1))

def rho(ma, rho0=1, gamma = 1.4):
  return rho0 * gd_isen(ma, gamma) ** (1/(gamma-1))

def kineticEnergyLossCoefficient(p0_out, p0_in, p_out=1, gamma=1.4):
  expon = (gamma-1)/gamma
  return 1- (1-(p_out/p0_out)**expon)/(1-(p_out/p0_in)**expon)

def totalPressureLossCoefficient(p0_out, p0_in, p_out, p_in, gamma=1.4):
  return (p0_in-p0_out)/(p0_in-p_in)

flux_mass     = lambda localMflux               :    (localMflux                          )
flux_momXP    = lambda localMflux, vX,  p       :    (localMflux*vX+p                     )
flux_momX     = lambda localMflux, vX           :    (localMflux*vX                       )
flux_momY     = lambda localMflux, vY           :    (localMflux*vY                       )
flux_enthalpy = lambda localMflux,  p, p0, gamma:    (localMflux*(p/p0)**((gamma-1)/gamma))
flux_entropy  = lambda localMflux, p0, p01, r   : -r*(localMflux*np.log(p0/p01)           )

def fluxesInt(data, p01, t, fluid):
  f = dict()

  lf = data['v_ax']*data['rho']

  f['i_m'] = np.trapz(flux_mass    (lf                                         ),  data['x'])/t
  f['i_f'] = np.trapz(flux_momXP   (lf, data['v_ax'],  data['p']                 ),  data['x'])/t
  f['i_a'] = np.trapz(flux_momX    (lf, data['v_ax']                             ),  data['x'])/t
  f['i_c'] = np.trapz(flux_momY    (lf, data['v_tan']                             ),  data['x'])/t
  f['i_h'] = np.trapz(flux_enthalpy(lf, data['p'] , data['p0'] , fluid['gamma']),  data['x'])/t
  f['i_s'] = np.trapz(flux_entropy (lf, data['p0'], p01        , fluid['r']    ),  data['x'])/t

  return f

def fluxesDisc(values, p01, fluid):
  f = dict()

  lf = values['v_ax']*values['rho']

  f['i_m'] = flux_mass    (lf)
  f['i_f'] = flux_momXP   (lf, values['v_ax'],  values['p'] )
  f['i_a'] = flux_momX    (lf, values['v_ax']      )
  f['i_c'] = flux_momY    (lf, values['v_tan']     )
  f['i_h'] = flux_enthalpy(lf, values['p'] , values['p0'] , fluid['gamma'])
  f['i_s'] = flux_entropy (lf, values['p0'], p01,           fluid['r']    )

  return f

def preproSPLEENdata(data: pd.DataFrame, 
                     theorOutletAngle=np.deg2rad(53.8),
                     pitch=32.950e-3,
                     r=8314.3/28.96,
                     gamma=1.4) -> pd.DataFrame:
  cp = gamma*r/(gamma-1)
  
  data.rename(columns={'y/g [-]'     :'x',
                       'd [deg]'     :'alpha_2',
                       'rho [kg/m^3]':'rho',
                       'V_ax [m/s]'  :'v_ax',
                       'P06/P01 [-]' :'p0/p01',
                       'Ps6/P01 [-]' :'p/p01'}, 
                       inplace=True)
  for k in data.keys():
    if ']' in k:
      data.drop(columns=k, inplace=True)

  data['x'] *= pitch
  data['alpha_2'] = np.deg2rad(data['alpha_2']) + theorOutletAngle

  data['p/p0'] = data['p/p01']/data['p0/p01']
  data['Ma'] = ma_is(data['p/p0'], 1, gamma)
  data['T'] = (data['v_ax']/np.cos(data['alpha_2'])/data['Ma'])**2/gamma/r
   
  a_t0 = data['T']+(data['v_ax']/np.cos(data['alpha_2']))**2/2/cp
  t0 = np.mean(a_t0)

  data['p']  = r*data['rho']*data['T']
  data['p0'] = data['p']/data['p/p0']
  data['p01']= data['p']/data['p/p01']

  data['v_tan'] = data['v_ax']*np.tan(data['alpha_2'])

  return data, t0, np.std(a_t0)
def preproKOBRAdata(data: pd.DataFrame, 
                     r=8314.3/28.96,
                     gamma=1.4) -> pd.DataFrame:
  cp = gamma*r/(gamma-1)
  
  data.rename(columns={'y/t'    :'x',
                       'a2'     :'alpha_2',
                       'T'      :'T',
                       'p2'     :'p',
                       'p02'    :'p0'}, 
                       inplace=True)
  data['p'] *= 1e3
  data['p0'] *= 1e3

  beta = data['p']/data['p0']
  a = (gamma-1)/gamma

  data['p01'] = data['p']/1000 * (1-(1-beta**a)/(1-data['dzeta']))**(-1/a)
  print(np.mean(data['p01']))
  data['alpha_2'] = np.deg2rad(data['alpha_2'])

  data['p/p0'] = data['p']/data['p0']
  data['Ma'] = ma_is(data['p/p0'], 1, gamma)
  t_over_to  = t(data['Ma'], gamma=gamma)
  t0 = np.mean(data['T']/t_over_to)

  data['rho']  = data['p']/(r*data['T'])

  data['v_ax']  = data['Ma']*np.sqrt(gamma*r*data['T'])*np.cos(data['alpha_2'])
  data['v_tan'] = data['v_ax']*np.tan(data['alpha_2'])

  return data, t0, np.std(data['T']/t_over_to)

def preproSCHREIBERdata(data: pd.DataFrame, 
                        r=8314.3/28.96,
                        gamma=1.4) -> pd.DataFrame:
  cp = gamma*r/(gamma-1)

  data.sort_values('Points:0', inplace=True)
  data['x'] = np.sqrt(data['Points:0']**2 + data['Points:1']**2 )  
  data['x'] -= data['x'].min()

  for k in data.keys():
    if 'Points' in k or 'U:2' in k:
      data.drop(columns=k, inplace=True)

  alpha_2 = np.empty(len(data.index))
  outletNorm = np.array([0.041866, -0.0370266]); outletNorm /= np.linalg.norm(outletNorm)
  for i in range(len(data.index)):
    velVec = np.array([data['U:0'][i], data['U:1'][i]]); velVec/=np.linalg.norm(velVec)
    alpha_2[i] = np.arccos(np.dot(outletNorm, velVec))
  
  data['alpha_2'] = alpha_2

  data['rho'] = data['p']/data['T']/r
  umag = np.sqrt(data['U:0']**2 + data['U:1']**2)
  data['Ma'] = umag /np.sqrt( (gamma*r*data['T']) )

  data['p0'] = data['p']/p(data['Ma'], 1, gamma)
  data['v_ax']  = umag*np.cos(data['alpha_2'])
  data['v_tan'] = umag*np.sin(data['alpha_2'])

  return data, 303.15

if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')