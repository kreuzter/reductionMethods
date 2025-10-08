#! /usr/bin/env python3

import numpy as np
from . import auxiliaryFunctions as aux

import matplotlib.pyplot as plt

class TraversingData:
  """
  Class for representation and manipulation with traversing data.
  """

  def __init__(self, data:dict, inlet:dict, uncertainties = {'p':0., 'p0':0., 'alpha':0.}, fluid = {'r':8314.3/28.96, 'gamma':1.4}):
    self.fluid = fluid
    if 'cp' not in self.fluid.keys(): self.fluid['cp'] = self.fluid['gamma']*self.fluid['r']/(self.fluid['gamma']-1)

    self.rawData = data
    self.rawData['T0'] = np.ones_like(data['p'])*inlet['T0']
    self.rawData['p01']= np.ones_like(data['p'])*inlet['p0']
    self.rawData['p1'] = np.ones_like(data['p'])*inlet['p' ]

    self.data_uncertainties = { v : np.ones_like(self.rawData['p'])*uncertainties[v] for v in uncertainties.keys()}
    self.dictiUncertainties_p_p0_alpha_T0 = {v:self.data_uncertainties[v] for v in ['p', 'p0', 'alpha', 'T0']}
    self.inlet   = inlet

    self.pitch = self.rawData['x'].max()-self.rawData['x'].min()
    self.averagingFunctions = self.functionsForAveraging()

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
    self.rawData = self.otherFrom_p_p0_alpha(self.rawData, self.data_uncertainties)
    self.rawData['loss_kin'], self.rawData['loss_tot_dynIn'], self.rawData['loss_tot_dynOut'], self.rawData['loss_tot_tot'] = self.getLosses(self.rawData) 

  def otherFrom_p_p0_alpha(self, dicti:dict, dictiUncertainties:dict):
    for variable in aux.from_p_p0_alpha_T0.keys():

      dicti[variable] = aux.from_p_p0_alpha_T0[variable](dicti['p'],dicti['p0'],dicti['alpha'], dicti['T0'], self.fluid)

      localUncs = dict()
      for variable2 in dictiUncertainties.keys():
        localDicti = { v : dicti[v] for v in dicti.keys()}
        localDicti[variable2] = dicti[variable2] + dictiUncertainties[variable2]
        localUncs[variable2] = aux.from_p_p0_alpha_T0[variable](localDicti['p'],localDicti['p0'],localDicti['alpha'], localDicti['T0'],self.fluid)-dicti[variable]

      try: 
        dictiUncertainties[variable] = np.linalg.norm(np.array( [localUncs[v] for v in localUncs.keys()] ), axis=0)  
      except:
        dictiUncertainties[variable] = np.linalg.norm(np.array( [localUncs[v] for v in localUncs.keys()] ))  
    
    return dicti

  def fluxes(self):
    if not (hasattr(self, 'trueFluxes')):
      print('Computing fluxes.')

      self.trueFluxes = dict()
      self.trueFluxesUncertainties = dict()
      for fluxName in aux.fluxesIntegrands.keys():
      
        self.trueFluxes[fluxName] = 1/self.pitch*np.trapezoid(aux.fluxesIntegrands[fluxName](self.rawData['p'],self.rawData['p0'],self.rawData['alpha'], self.rawData['T0'], self.rawData['p01'], self.fluid), self.rawData['x'])
  
        localUncs = dict()
        for variable2 in self.dictiUncertainties_p_p0_alpha_T0.keys():
          localDicti = { v : self.rawData[v] for v in self.rawData.keys()}
          localDicti[variable2] = self.rawData[variable2] + self.dictiUncertainties_p_p0_alpha_T0[variable2]
          localUncs[variable2] = 1/self.pitch*np.trapezoid(aux.fluxesIntegrands[fluxName](localDicti['p'],localDicti['p0'],localDicti['alpha'], localDicti['T0'], localDicti['p01'], self.fluid), localDicti['x']) -self.trueFluxes[fluxName]
  
        self.trueFluxesUncertainties[fluxName] = np.linalg.norm(np.array( [localUncs[v] for v in localUncs.keys()] ), axis=0)  

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

  def reductionMethod(func, additionalParameter = None):
    def wrapper(self, additionalParameter):
      result = func(self, additionalParameter)
      result = self.getAdditionalProperties(result)
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
  
  @reductionMethod
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
    return reduced

  @reductionMethod
  def reduction_vzlu(self, additionalParameter):
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

  def weighted(self, how, values, p, p0, alpha, T0, p01, fluid):
    assert how in ['mass', 'massFlux', 'area']
    if how == 'massFlux' or how == 'mass':
      return np.trapezoid(values*aux.fluxesIntegrands['I_M'](p, p0, alpha, T0, p01, fluid), self.rawData['x'] ) / np.trapezoid(aux.fluxesIntegrands['I_M'](p, p0, alpha, T0, p01, fluid), self.rawData['x'] )
    elif how == 'area':
      return np.trapezoid( values, self.rawData['x'] )/self.pitch
    
  angleAsArctanOfFluxes = lambda self, p, p0, alpha, T0, p01, fluid: np.arctan( np.trapezoid(aux.fluxesIntegrands['I_C'](p, p0, alpha, T0, p01, fluid), self.rawData['x'] )/np.trapezoid(aux.fluxesIntegrands['I_A'](p, p0, alpha, T0, p01, fluid), self.rawData['x'] ) )

  def textInfoAboutAveraging(self, method):
    match method:
      case 'massFlux':
        return {
                 'method_name' : 'Mass Flux Weighted Averaging',
                 'method_abbr' : r'$\dot{m}$',
                }
      case 'area':
        return {
                  'method_name' : 'Area Weighted Averaging',
                  'method_abbr' : r'$A$',
                }
      case 'momentum':
        return {
                  'method_name' : 'Momentum Weighted Averaging',
                  'method_abbr' : 'MOM',
                }
      case 'enthalpy':
        return {
                  'method_name' : 'Enthalpy Weighted Averaging',
                  'method_abbr' : r'$h$',
                }
      case 'entropy':
        return {
                  'method_name' : 'Entropy Weighted Averaging',
                  'method_abbr' : r'$s$',
                }

  def functionsForAveraging(self):
    dicti = { 'area' : {
      'p'     : lambda p, p0, alpha, T0, p01, fluid: self.weighted('area',     p, p, p0, alpha, T0, p01, fluid),
      'p0'    : lambda p, p0, alpha, T0, p01, fluid: self.weighted('area',    p0, p, p0, alpha, T0, p01, fluid),
      'alpha' : lambda p, p0, alpha, T0, p01, fluid: self.weighted('area', alpha, p, p0, alpha, T0, p01, fluid),
      'T0'    : lambda p, p0, alpha, T0, p01, fluid: T0[0]
    } }
    
    dicti['massFlux'] = {
      'p'     : lambda p, p0, alpha, T0, p01, fluid: self.weighted('massFlux',     p, p, p0, alpha, T0, p01, fluid),
      'p0'    : lambda p, p0, alpha, T0, p01, fluid: self.weighted('massFlux',    p0, p, p0, alpha, T0, p01, fluid),
      'alpha' : lambda p, p0, alpha, T0, p01, fluid: self.weighted('massFlux', alpha, p, p0, alpha, T0, p01, fluid),
      'T0'    : lambda p, p0, alpha, T0, p01, fluid: T0[0]
    }

    dicti['momentum'] = {
      'p'     : lambda p, p0, alpha, T0, p01, fluid: self.weighted('area',  p, p, p0, alpha, T0, p01, fluid),
      'p0'    : lambda p, p0, alpha, T0, p01, fluid: self.weighted('area',  p, p, p0, alpha, T0, p01, fluid)*(1-(fluid['gamma']-1)*self.weighted('massFlux', aux.from_p_p0_alpha_T0['v_mag'](p, p0, alpha, T0, fluid), p, p0, alpha, T0, p01, fluid)**2/(2*fluid['gamma']*fluid['r']*T0[0]))**(fluid['gamma']/(1-fluid['gamma'])),
      'alpha' : lambda p, p0, alpha, T0, p01, fluid: self.angleAsArctanOfFluxes(p, p0, alpha, T0, p01, fluid),
      'T0'    : lambda p, p0, alpha, T0, p01, fluid: T0[0]
    }

    dicti['enthalpy'] = {
      'p'     : lambda p, p0, alpha, T0, p01, fluid: self.weighted('area',  p, p, p0, alpha, T0, p01, fluid),
      'p0'    : lambda p, p0, alpha, T0, p01, fluid: self.weighted('area',  p, p, p0, alpha, T0, p01, fluid)*((self.weighted('massFlux', aux.from_p_p0_alpha_T0['T'](p, p0, alpha, T0, fluid), p, p0, alpha, T0, p01, fluid))/T0[0])**(fluid['gamma']/(1-fluid['gamma'])),
      'alpha' : lambda p, p0, alpha, T0, p01, fluid: self.angleAsArctanOfFluxes(p, p0, alpha, T0, p01, fluid),
      'T0'    : lambda p, p0, alpha, T0, p01, fluid: T0[0]
    }

    dicti['entropy'] = {
      'p'     : lambda p, p0, alpha, T0, p01, fluid: self.weighted('area',  p, p, p0, alpha, T0, p01, fluid),
      'p0'    : lambda p, p0, alpha, T0, p01, fluid: np.exp(self.weighted('massFlux', np.log(p0), p, p0, alpha, T0, p01, fluid)),
      'alpha' : lambda p, p0, alpha, T0, p01, fluid: self.angleAsArctanOfFluxes(p, p0, alpha, T0, p01, fluid),
      'T0'    : lambda p, p0, alpha, T0, p01, fluid: T0[0]
    }

    return dicti

  @reductionMethod
  def reduction_universalAveraging(self, kind : str ):
    reduced = { variable : self.averagingFunctions[kind][variable](self.rawData['p'], self.rawData['p0'], self.rawData['alpha'], self.rawData['T0'], self.rawData['p01'], self.fluid) for variable in self.averagingFunctions[kind].keys()}
    reduced['uncertainties'] = dict()
    for variable in self.averagingFunctions[kind].keys(): 
      localUncs = dict()
      
      for variable2 in self.dictiUncertainties_p_p0_alpha_T0.keys():
        localDicti = { v : self.rawData[v] for v in self.rawData.keys()}
        localDicti[variable2] = self.rawData[variable2] + self.dictiUncertainties_p_p0_alpha_T0[variable2]
        
        localUncs[variable2] = self.averagingFunctions[kind][variable](localDicti['p'], localDicti['p0'], localDicti['alpha'], localDicti['T0'], localDicti['p01'], self.fluid) - reduced[variable]
      
      reduced['uncertainties'][variable] = np.linalg.norm(np.array( [localUncs[v] for v in localUncs.keys()] ))  
      
    
    reduced.update(self.textInfoAboutAveraging(kind))
    reduced = self.otherFrom_p_p0_alpha(reduced, reduced['uncertainties'])
    return reduced

if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')