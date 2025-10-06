#! /usr/bin/env python3

import numpy as np
from . import auxiliaryFunctions as aux

class TraversingData:
  """
  Class for representation and manipulation with traversing data.
  """

  def __init__(self, data:dict, inlet:dict, fluid = {'r':8314.3/28.96, 'gamma':1.4}):
    self.fluid = fluid
    if 'cp' not in self.fluid.keys(): self.fluid['cp'] = self.fluid['gamma']*self.fluid['r']/(self.fluid['gamma']-1)

    self.rawData = data
    self.inlet   = inlet

    self.pitch = self.rawData['x'].max()-self.rawData['x'].min()

    self.prepro()

  def reduceByAll(self, mm = True):
    toRet = {}
    for name in dir(self):
      if name.startswith('reduction_'):
        method = getattr(self, name)
        m = method() if name != 'reduction_momentumMethod' else method(mm)
        m['alpha_d'] = np.rad2deg(m['alpha'])
        toRet[name] = m
    return toRet   

  def prepro(self):
    """Preprocess traversing data."""

    self.rawData['alpha_d'] = np.rad2deg(self.rawData['alpha'])
    self.rawData = self.otherFrom_p_p0_alpha(self.rawData)
    self.rawData['loss_kin'], self.rawData['loss_tot_dynIn'], self.rawData['loss_tot_dynOut'], self.rawData['loss_tot_tot'] = self.getLosses(self.rawData) 

  def otherFrom_p_p0_alpha(self, dicti:dict):
    dicti['M']     = aux.ma_is(dicti['p'], dicti['p0'], self.fluid['gamma'])
    dicti['T']     = aux.t(dicti['M'], self.inlet['T0'], self.fluid['gamma'])
    dicti['rho']   = aux.rho(dicti['M'], dicti['p0']/self.inlet['T0']/self.fluid['r'], self.fluid['gamma'])
    dicti['v_mag'] = dicti['M'] * np.sqrt( self.fluid['gamma'] * self.fluid['r'] * dicti['T'] )
    dicti['v_x']   = dicti['v_mag'] * np.cos(dicti['alpha'])
    dicti['v_y']   = dicti['v_mag'] * np.sin(dicti['alpha'])

    return dicti

  def fluxes(self):
    if not (hasattr(self, 'trueFluxes')):
      print('Computing fluxes.')
      self.trueFluxes = aux.fluxesInt(self.rawData, self.inlet['p0'], self.pitch, self.fluid)

    return self.trueFluxes

  def trueTotalPressureLossCoefficient_totIn(self):
    fluxes = self.fluxes()
    ds = fluxes['I_S']/fluxes['I_M']
    coeff = 1 - np.exp(-ds/self.fluid['r'])
    return coeff

  def trueTotalPressureLossCoefficient_dynIn(self):
    fluxes = self.fluxes()
    ds = fluxes['I_S']/fluxes['I_M']
    coeff = self.inlet['p0']/(self.inlet['p0']-self.inlet['p'])*(1 - np.exp(-ds/self.fluid['r']))
    return coeff

  def checkEOS(self, dicti:dict):
    p = self.fluid['r']*dicti['T']*dicti['rho']
    satisfied = np.isclose(dicti['p'], p, atol=1)
    return satisfied, p-dicti['p']
  
  def getLosses(self, dicti:dict):
    return aux.losses(dicti['p'],self.inlet['p'],dicti['p0'], self.inlet['p0'])
  
  def getEntropyIncrease(self, dicti:dict):
    s = self.fluxes()['I_S']/self.fluxes()['I_M']
    return ((aux.s(dicti['p0'], self.inlet['p0'], self.fluid['r']) - s)/s)*100
  
  def checkTotalTemperature(self, dicti:dict):
    #t0 = dicti['T'] + (dicti['v_x']**2 + dicti['v_y']**2)/2/self.fluid['cp']
    t0 = dicti['T']/aux.t(dicti['M'], 1, self.fluid['gamma'])
    satisfied = np.isclose(self.inlet['T0'], t0, atol=0.1)
    return satisfied, t0-self.inlet['T0']
  
  def getFluxes(self, dicti:dict):
    return aux.fluxesDisc(dicti, self.inlet['p0'], self.fluid) 
  
  def getAdditionalProperties(self, dicti:dict):
    dicti['fluxes'] = self.getFluxes(dicti)
    dicti['loss_kin'], dicti['loss_tot_dynIn'], dicti['loss_tot_dynOut'], dicti['loss_tot_tot'] = self.getLosses(dicti)
    dicti['entropy_increase'] = self.getEntropyIncrease(dicti)
    dicti['EOSsatisfied'] = self.checkEOS(dicti)
    dicti['T0Satisfied'] = self.checkTotalTemperature(dicti)

    return dicti

  def getReducedFrom_p_p0_alpha(self, dicti:dict):
    dicti = self.otherFrom_p_p0_alpha(dicti)
    dicti = self.getAdditionalProperties(dicti)
    
    return dicti

  def reductionMethod(func):
    def wrapper(self):
      result = func(self)
      result = self.getAdditionalProperties(result)
      return result
    return wrapper

  def averaging(func):
    def wrapper(self):
      result = func(self)
      result = self.otherFrom_p_p0_alpha(result)
      return result
    return wrapper
  
  def weightedAverage(self, which:str, what:str):
    if which == 'mass':
      return np.trapz(self.rawData[what]*self.rawData['v_x']*self.rawData['rho'], self.rawData['x'])/(self.fluxes()['I_M']*self.pitch)
    elif which == 'area':
      return np.trapz(self.rawData[what], self.rawData['x'])/self.pitch
    else:
      NotImplementedError
    
  def plotRawData(self, vars, figsize = (12,6), ylabels = None, figax = None, legenPrepend = '_', legendkwargs = {}, cycLinestyle=['-', '--', ':', '-.'],
                  rcParams_user = {}):
    if ylabels == None: ylabels = [None]*len(vars)

    import matplotlib.pyplot as plt
    from cycler import cycler
    
    rcParams_default = {
      "text.usetex": True,
      "font.family": "Times",
      "font.serif" : "Times New Roman",
      "font.size"  : 12
    }

    plt.rcParams.update(rcParams_default)
    plt.rcParams.update(rcParams_user)

    default_cycler = (cycler(linestyle=cycLinestyle))

    plt.rc('lines', linewidth=0.75, color = 'k')
    plt.rc('axes', prop_cycle=default_cycler)

    labels = {
      'p'               : [r'$p$'                                  , 'Pa'        ],
      'p0'              : [r'$p_0$'                                , 'Pa'        ],
      'M'               : ['M'                                     , '1'         ],
      'rho'             : [r'$\rho$'                               , r'kg/m$^3$' ],
      'T'               : [r'$T$'                                  , 'K'         ],
      'alpha'           : [r'$\alpha$'                             , '°'         ], 
      'v_mag'           : [r'$||\mathbf{w}||$'                     , 'm/s'       ],
      'v_x'             : [r'$w_x$'                                , 'm/s'       ],
      'v_y'             : [r'$w_y$'                                , 'm/s'       ],
      'loss_kin'        : [r'$\xi$'                                , '1'         ],
      'loss_tot_dynIn'  : [r'$\omega_{\, \mathrm{rel. to } \, 1}$' , '1'         ],
      'loss_tot_dynOut' : [r'$\omega_{\, \mathrm{rel. to } \, 2}$' , '1'         ],
      'loss_tot_tot'    : [r'$\zeta$'                              , '1'         ], 
    }

    if figax == None:
      fig, ax = plt.subplots(len(vars), figsize=figsize, sharex=True)
    else:
      fig, ax = figax
    x = (self.rawData['x'] - self.rawData['x'].min())/(self.rawData['x'].max() - self.rawData['x'].min())
    for i in range(len(vars)):
      for j in range(len(vars[i])):
        ax[i].plot(x, self.rawData[vars[i][j]] if vars[i][j] != 'alpha' else np.rad2deg(self.rawData[vars[i][j]]), 
                   label = (len(vars[i]) <= 1 and legenPrepend == '_')*'_'+ (legenPrepend != '_')*legenPrepend +(len(vars[i]) > 1)*(labels[vars[i][j]][0])
                   )
      ax[i].grid(True)
      ax[i].set_ylabel(
        f'{labels[vars[i][0]][0]} [{labels[vars[i][0]][1]}]' if ylabels[i] == None else ylabels[i]
      )
      ax[i].set_xlim([0,1])

      if (len(vars[i]) > 1 or (legenPrepend != '_' and i <1)): 
        ax[i].legend(fancybox = False, frameon=True, edgecolor = 'k', framealpha = 1, 
                     **legendkwargs)

    ax[-1].set_xlabel(r'$Y$ [1]')
    fig.align_ylabels()
    fig.tight_layout()
    
    return fig, ax
  
  def reduction_momentumMethod(self, normal = True):

    reduced = {
      'method_name' : 'Momentum Method',
      'method_abbr' : 'MM'
    }
    fluxes = self.fluxes()

    d = fluxes['I_F']**2-4*(1-self.fluid['r']/2/self.fluid['cp'])*(self.inlet['T0']*self.fluid['r']*fluxes['I_M']**2 - self.fluid['r']/2/self.fluid['cp'] * fluxes['I_C']**2)
    z = (fluxes['I_F']+(-1)**normal*np.sqrt(d))/2/(1-self.fluid['r']/2/self.fluid['cp'])

    reduced.update({
      'p'    : fluxes['I_F']-z,
      'rho'  : fluxes['I_M']**2/z,
      'v_y'  : fluxes['I_C']/fluxes['I_M']
    })

    reduced['T'] = reduced['p']/reduced['rho']/self.fluid['r']
    reduced['v_x'] = fluxes['I_M']/reduced['rho']
    reduced.update({
      'alpha':np.arctan2(reduced['v_y'], reduced['v_x']),
      'M':np.sqrt( (2*self.fluid['cp']*(self.inlet['T0']-reduced['T'])) / (self.fluid['gamma']*self.fluid['r']*reduced['T']) ),
    })
    p_over_p0 = aux.p(reduced['M'])
    reduced['p0'] = reduced['p']/p_over_p0 

    reduced['loss_kin'], reduced['loss_tot_dynIn'], reduced['loss_tot_dynOut'], reduced['loss_tot_tot'] = self.getLosses(reduced)

    reduced['fluxes'] = self.getFluxes(reduced)
    reduced['entropy_increase'] = self.getEntropyIncrease(reduced)

    reduced['EOSsatisfied'] = self.checkEOS(reduced)
    reduced['T0Satisfied'] = self.checkTotalTemperature(reduced)
    
    return reduced

  @reductionMethod
  def reduction_vzlu(self):
    fluxes = self.fluxes()

    reduced = {
      'method_name' : 'Strictly Conservative',
      'method_abbr' : 'SC',
      'p0':self.inlet['p0']*np.exp(-1/self.fluid['r']*fluxes['I_S']/fluxes['I_M']),
      'T' :self.inlet['T0']*fluxes['I_H']/fluxes['I_M'],
      'alpha':np.arctan2(fluxes['I_C'],fluxes['I_A']),
    }
    reduced['p']    = reduced['p0']*(fluxes['I_H']/fluxes['I_M'])**(self.fluid['gamma']/(self.fluid['gamma']-1))

    reduced['M']    = aux.ma_is(reduced['p'], reduced['p0'])
    reduced['v_x']  = fluxes['I_A']/fluxes['I_M']
    reduced['v_y']  = fluxes['I_C']/fluxes['I_M']
    reduced['rho']  =  fluxes['I_M']/reduced['v_x']

    '''reduced['loss_kin'], reduced['loss_tot_dynIn'], reduced['loss_tot_dynOut'], reduced['loss_tot_tot'] = self.getLosses(reduced)

    reduced['fluxes'] = self.getFluxes(reduced)
    reduced['entropy_increase'] = self.getEntropyIncrease(reduced)
    reduced['EOSsatisfied'] = self.checkEOS(reduced)
    reduced['T0Satisfied'] = self.checkTotalTemperature(reduced)'''

    return reduced

  @reductionMethod
  @averaging
  def reduction_areaDirect(self):
    reduced = {
      'method_name' : 'Area Weighted Averaging',
      'method_abbr' : r'$A$',
    }

    for v in ['p','p0','alpha']: 
      reduced[v] = self.weightedAverage('area', v)
    
    return reduced

  @reductionMethod
  @averaging
  def reduction_massFluxDirect(self):
    reduced = {
      'method_name' : 'Mass Flux Weighted Averaging',
      'method_abbr' : r'$\dot{m}$',
    }

    for v in ['p','p0','alpha']: 
      reduced[v] = self.weightedAverage('mass', v)
    
    return reduced

  @reductionMethod
  @averaging
  def reduction_momentum(self):
    reduced = {
      'method_name' : 'Momentum Weighted Averaging',
      'method_abbr' : 'MOM',
    }

    reduced['p'] = self.weightedAverage('area', 'p')
    
    for v in ['v_x','v_y', 'v_mag']: 
      reduced[v] = self.weightedAverage('mass', v)
    
    t = self.inlet['T0'] - reduced['v_mag']**2/2/self.fluid['cp']
    ma= reduced['v_mag']/np.sqrt(self.fluid['r']*self.fluid['gamma']*t)

    p2p0 = aux.p(ma, 1, self.fluid['gamma'])

    reduced['p0'] = reduced['p']/p2p0

    reduced['alpha'] = np.arctan2(reduced['v_y'], reduced['v_x'])
    del reduced['v_x'], reduced['v_y'], reduced['v_mag']

    return reduced
  
  @reductionMethod
  @averaging
  def reduction_enthalpy(self):
    reduced = {
      'method_name' : 'Enthalpy Weighted Averaging',
      'method_abbr' : r'$h$',
    }

    reduced['p'] = self.weightedAverage('area', 'p')
    raw_h = self.fluid['cp']*self.inlet['T0']*(self.rawData['p']/self.rawData['p0'])**((self.fluid['gamma']-1)/self.fluid['gamma'])

    for v in ['v_x','v_y']: 
      reduced[v] = self.weightedAverage('mass', v)

    reduced_h = np.trapz(raw_h*self.rawData['v_x']*self.rawData['rho'], self.rawData['x'])/(self.fluxes()['I_M']*self.pitch)
    reduced['p0'] = reduced['p'] * (reduced_h/self.fluid['cp']/self.inlet['T0'])**(self.fluid['gamma']/(1-self.fluid['gamma']))
    
    reduced['alpha'] = np.arctan2(reduced['v_y'], reduced['v_x'])
    del reduced['v_x'], reduced['v_y']

    return reduced

  @reductionMethod
  @averaging
  def reduction_entropy(self):
    reduced = {
      'method_name' : 'Entropy Weighted Averaging',
      'method_abbr' : r'$s$',
    }

    reduced['p'] = self.weightedAverage('area', 'p')

    for v in ['v_x','v_y']: 
      reduced[v] = self.weightedAverage('mass', v)

    reduced['p0'] = np.exp(np.trapz(np.log(self.rawData['p0'])*self.rawData['v_x']*self.rawData['rho'], self.rawData['x'])/(self.fluxes()['I_M']*self.pitch))
    
    reduced['alpha'] = np.arctan2(reduced['v_y'], reduced['v_x'])
    del reduced['v_x'], reduced['v_y']
    
    return reduced


if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')