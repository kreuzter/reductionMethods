#! /usr/bin/env python3

import numpy as np
from . import auxiliaryFunctions as aux

import matplotlib.pyplot as plt

class TraversingData:
  """
  Class for representation and manipulation with traversing data.
  """

  def __init__(self, data:dict, inlet:dict, uncertainties:dict, fluid = {'r':8314.3/28.96, 'gamma':1.4}):
    self.fluid = fluid
    if 'cp' not in self.fluid.keys(): self.fluid['cp'] = self.fluid['gamma']*self.fluid['r']/(self.fluid['gamma']-1)

    self.rawData = data
    self.inlet   = inlet

    self.measuredVariables = {
      'M'    : lambda values : aux.ma_is(values['p'], values['p0'], self.fluid['gamma']),
      'T'    : lambda values : aux.t(aux.ma_is(values['p'], values['p0'], self.fluid['gamma']), values['T0'], self.fluid['gamma']),
      'rho'  : lambda values : aux.rho(aux.ma_is(values['p'], values['p0'], self.fluid['gamma']), values['p0']/values['T0']/self.fluid['r'], self.fluid['gamma']),
      'v_mag': lambda values : aux.ma_is(values['p'], values['p0'], self.fluid['gamma']) * np.sqrt( self.fluid['gamma'] * self.fluid['r'] * aux.t(aux.ma_is(values['p'], values['p0'], self.fluid['gamma']), values['T0'], self.fluid['gamma'])), 
      'v_x'  : lambda values : aux.ma_is(values['p'], values['p0'], self.fluid['gamma']) * np.sqrt( self.fluid['gamma'] * self.fluid['r'] * aux.t(aux.ma_is(values['p'], values['p0'], self.fluid['gamma']), values['T0'], self.fluid['gamma'])) * np.cos(values['alpha']),
      'v_y'  : lambda values : aux.ma_is(values['p'], values['p0'], self.fluid['gamma']) * np.sqrt( self.fluid['gamma'] * self.fluid['r'] * aux.t(aux.ma_is(values['p'], values['p0'], self.fluid['gamma']), values['T0'], self.fluid['gamma'])) * np.sin(values['alpha'])
    }

    self.fluxesIntegrands = {
      'I_M' : lambda values :                   self.measuredVariables['v_x'](values)    * self.measuredVariables['rho'](values),
      'I_A' : lambda values :                   self.measuredVariables['v_x'](values)**2 * self.measuredVariables['rho'](values),
      'I_F' : lambda values :                   self.measuredVariables['v_x'](values)**2 * self.measuredVariables['rho'](values) + values['p'],
      'I_C' : lambda values :                   self.measuredVariables['v_x'](values)    * self.measuredVariables['rho'](values) * self.measuredVariables['v_y'](values) ,
      'I_H' : lambda values :                   self.measuredVariables['v_x'](values)    * self.measuredVariables['rho'](values) * (values['p']/values['p0'])**((self.fluid['gamma']-1)/self.fluid['gamma']),
      'I_S' : lambda values : -self.fluid['r']*(self.measuredVariables['v_x'](values)    * self.measuredVariables['rho'](values) *np.log(values['p0']/values['p01'])),
    }
     
    self.fluxesIntegrals = { 
      'I_M' : lambda values : 1/self.pitch*np.trapezoid(self.fluxesIntegrands['I_M'](values), self.rawData['x']),
      'I_A' : lambda values : 1/self.pitch*np.trapezoid(self.fluxesIntegrands['I_A'](values), self.rawData['x']),
      'I_F' : lambda values : 1/self.pitch*np.trapezoid(self.fluxesIntegrands['I_F'](values), self.rawData['x']),
      'I_C' : lambda values : 1/self.pitch*np.trapezoid(self.fluxesIntegrands['I_C'](values), self.rawData['x']),
      'I_H' : lambda values : 1/self.pitch*np.trapezoid(self.fluxesIntegrands['I_H'](values), self.rawData['x']),
      'I_S' : lambda values : 1/self.pitch*np.trapezoid(self.fluxesIntegrands['I_S'](values), self.rawData['x'])
    }   

    self.rawData['T0'] = np.ones_like(data['p'])*inlet['T0']
    self.rawData['p01']= np.ones_like(data['p'])*inlet['p0']
    self.rawData['p1'] = np.ones_like(data['p'])*inlet['p' ]

    self.rawData['alpha_d'] = np.rad2deg(self.rawData['alpha'])

    self.givenUncertainties = uncertainties
    self.rawDataUncertainties = { v : np.ones_like(self.rawData['p'])*uncertainties[v] for v in uncertainties.keys()}
    self.rawData = self.otherVariables_fromMeasured(self.rawData, self.rawDataUncertainties)
    
    self.pitch = self.rawData['x'].max()-self.rawData['x'].min()
    self.averagingFunctions = self.functionsForAveraging()   

  def reduceByAll(self, mm = True):
    toRet = {}
    for name in dir(self):
      if name.startswith('reduction_'):
        method = getattr(self, name)
        m = method() if name != 'reduction_momentumMethod' else method(mm)
        m['alpha_d'] = np.rad2deg(m['alpha'])
        toRet[name] = m
    return toRet   

  def otherVariables_fromMeasured(self, dicti:dict, dictiUncertainties:dict):
    
    dicti.update({ variable : self.measuredVariables[variable](dicti) for variable in self.measuredVariables.keys()})
    dictiUncertainties.update(self.computeUncertainties(self.measuredVariables, dicti, dicti))
    
    return dicti

  def integralFluxes(self):
    if not (hasattr(self, 'trueFluxes')):
      self.trueFluxes = { fluxName : self.fluxesIntegrals[fluxName](self.rawData) for fluxName in self.fluxesIntegrands.keys()}
      self.trueFluxesUncertainties = self.computeUncertainties(self.fluxesIntegrals, self.trueFluxes, self.rawData)       
    return self.trueFluxes, self.trueFluxesUncertainties

  def meanFluxes(self, dicti:dict):
    fluxes = { fluxName : self.fluxesIntegrands[fluxName](dicti) for fluxName in self.fluxesIntegrands.keys()}
    fluxesUncertainties = self.computeUncertainties(self.fluxesIntegrands, fluxes, dicti)
    return fluxes, fluxesUncertainties

  def trueTotalPressureLossCoefficient_totIn(self):
    fluxes, _ = self.integralFluxes()
    ds = fluxes['I_S']/fluxes['I_M']
    coeff = 1 - np.exp(-ds/self.fluid['r'])
    return coeff

  def trueTotalPressureLossCoefficient_dynIn(self):
    fluxes, _ = self.integralFluxes()
    ds = fluxes['I_S']/fluxes['I_M']
    coeff = self.inlet['p0']/(self.inlet['p0']-self.inlet['p'])*(1 - np.exp(-ds/self.fluid['r']))
    return coeff

  def checkEoS(self, dicti:dict):
    p = self.fluid['r']*dicti['T']*dicti['rho']
    satisfied = np.isclose(dicti['p'], p, atol=1)
    return satisfied, p-dicti['p']
  
  def getLosses(self, dicti:dict):
    losses = aux.losses(dicti['p'],self.inlet['p'],dicti['p0'], self.inlet['p0'])

    lossesUncertainties = dict()
    for lossName in aux.losses.keys():
    
      losses[lossName] = aux.losses[lossName](dicti['p'], dicti['p1'],dicti['p0'], dicti['p01'], self.fluid['gamma'])

      localUncs = dict()
      for variable2 in self.rawDataUncertainties.keys():
        localDicti = { v : dicti[v] for v in dicti.keys()}
        localDicti[variable2] = dicti[variable2] + self.rawDataUncertainties[variable2][0]
        localUncs[variable2] = aux.lossesIntegrands[lossName](localDicti['p'],localDicti['p0'],localDicti['alpha'], localDicti['T0'], localDicti['p01'], self.fluid) -losses[lossName]

      lossesUncertainties[lossName] = np.linalg.norm(np.array( [localUncs[v] for v in localUncs.keys()] ), axis=0)  
    return losses, lossesUncertainties
  
  def getEntropyIncrease(self, dicti:dict):
    s = self.integralFluxes()[0]['I_S']/self.integralFluxes()[0]['I_M']
    print(s)
    return ((aux.s(dicti['p0'], self.inlet['p0'], self.fluid['r']) - s)/s)*100

  def reductionMethod(func, additionalParameter = None):
    def wrapper(self, additionalParameter):
      result = func(self, additionalParameter)
      #result = self.getAdditionalProperties(result)
      return result
    return wrapper

  def weightedAverage(self, which:str, what:str):
    if which == 'mass':
      return np.trapz(self.rawData[what]*self.rawData['v_x']*self.rawData['rho'], self.rawData['x'])/(self.integralFluxes()[0]['I_M']*self.pitch)
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
    lambdas = dict()
    lambdas['d'] = lambda values : self.fluxesIntegrals['I_F'](values)**2-4*(1-self.fluid['r']/2/self.fluid['cp'])*(values['T0'][0]*self.fluid['r']*self.fluxesIntegrals['I_M'](values)**2 - self.fluid['r']/2/self.fluid['cp'] * self.fluxesIntegrals['I_C'](values)**2)
    lambdas['z'] = lambda values: (self.fluxesIntegrals['I_F'](values)+(-1)**normal*np.sqrt(lambdas['d'](values)))/2/(1-self.fluid['r']/2/self.fluid['cp'])

    lambdas.update({
      'p'    : lambda values : self.fluxesIntegrals['I_F'](values)-lambdas['z'](values),
      'rho'  : lambda values : self.fluxesIntegrals['I_M'](values)**2/lambdas['z'](values),
      'v_y'  : lambda values : self.fluxesIntegrals['I_C'](values)/self.fluxesIntegrals['I_M'](values),
      'T0'    : lambda values: values['T0'][0],
      'p01'   : lambda values: values['p01'][0],
      'p1'    : lambda values: values['p1'][0]
    })

    lambdas['T']   = lambda values : lambdas['p'](values)/lambdas['rho'](values)/self.fluid['r']
    lambdas['v_x'] = lambda values : self.fluxesIntegrals['I_M'](values)/lambdas['rho'](values)
    
    lambdas.update({
      'alpha': lambda values : np.arctan2(lambdas['v_y'](values), lambdas['v_x'](values)),
      'M':     lambda values : np.sqrt( (2*self.fluid['cp']*(values['T0'][0]-lambdas['T'](values))) / (self.fluid['gamma']*self.fluid['r']*lambdas['T'](values)) ),
    })
    lambdas['p_over_p0'] = lambda values: aux.p(lambdas['M'](values))
    lambdas['p0'] = lambda values: lambdas['p'](values)/lambdas['p_over_p0'](values)    

    reduced.update({ variable : lambdas[variable](self.rawData) for variable in lambdas.keys()})

    reduced['uncertainties'] = self.computeUncertainties(lambdas, reduced, self.rawData)

    reduced['fluxes'], reduced['fluxesUncertainties'] = self.meanFluxes(reduced)
    reduced = self.otherVariables_fromMeasured(reduced, reduced['uncertainties'])    

    return reduced

  @reductionMethod
  def reduction_vzlu(self, additionalParameter = None):
    
    lambdas = {
      'p0' :   lambda values : values['p01'][0]*np.exp(-1/self.fluid['r']*self.fluxesIntegrals['I_S'](values)/self.fluxesIntegrals['I_M'](values)),
      'T'  :   lambda values : values['T0'][0]*self.fluxesIntegrals['I_H'](values)/self.fluxesIntegrals['I_M'](values),
      'alpha': lambda values : self.angleAsArctanOfFluxes(values),
      'T0'    : lambda values: values['T0'][0],
      'p01'   : lambda values: values['p01'][0],
      'p1'    : lambda values: values['p1'][0]
    }
    lambdas['p'] = lambda values : lambdas['p0'](values)*(self.fluxesIntegrals['I_H'](values)/self.fluxesIntegrals['I_M'](values))**(self.fluid['gamma']/(self.fluid['gamma']-1))
    lambdas['M'] = lambda values : aux.ma_is(lambdas['p'](values), lambdas['p0'](values), self.fluid['gamma'])
    lambdas['v_x']  = lambda values: self.fluxesIntegrals['I_A'](values)/self.fluxesIntegrals['I_M'](values)
    lambdas['v_y']  = lambda values: self.fluxesIntegrals['I_C'](values)/self.fluxesIntegrals['I_M'](values)
    lambdas['rho']  = lambda values: self.fluxesIntegrals['I_M'](values)/lambdas['v_x'](values)

    reduced = {
      'method_name' : 'Strictly Conservative',
      'method_abbr' : 'SC'
    }

    reduced.update({ variable : lambdas[variable](self.rawData) for variable in lambdas.keys()})

    reduced['uncertainties'] = self.computeUncertainties(lambdas, reduced, self.rawData)

    reduced['fluxes'], reduced['fluxesUncertainties'] = self.meanFluxes(reduced)
    reduced = self.otherVariables_fromMeasured(reduced, reduced['uncertainties'])
    return reduced

  def weighted(self, how, what, values):
    assert how in ['mass', 'massFlux', 'area']
    if how == 'massFlux' or how == 'mass':
      return np.trapezoid(what*self.fluxesIntegrands['I_M'](values), self.rawData['x'] ) / np.trapezoid(self.fluxesIntegrands['I_M'](values), self.rawData['x'] )
    elif how == 'area':
      return np.trapezoid(what, self.rawData['x'] )/self.pitch
  
  def angleAsArctanOfFluxes(self, values): 
    return np.arctan( self.fluxesIntegrals['I_C'](values)/self.fluxesIntegrals['I_A'](values) )

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
                  'method_name' : 'Momentum Averaging',
                  'method_abbr' : 'MOM',
                }
      case 'enthalpy':
        return {
                  'method_name' : 'Enthalpy Averaging',
                  'method_abbr' : r'$h$',
                }
      case 'entropy':
        return {
                  'method_name' : 'Entropy Averaging',
                  'method_abbr' : r'$s$',
                }

  def functionsForAveraging(self):
    dicti = { 'area' : {
      'p'     : lambda values: self.weighted('area',     values['p'], values),
      'p0'    : lambda values: self.weighted('area',    values['p0'], values),
      'alpha' : lambda values: self.weighted('area', values['alpha'], values),
      'T0'    : lambda values: values['T0'][0],
      'p01'   : lambda values: values['p01'][0],
      'p1'    : lambda values: values['p1'][0]
    } }
    
    dicti['massFlux'] = {
      'p'     : lambda values: self.weighted('massFlux',     values['p'], values),
      'p0'    : lambda values: self.weighted('massFlux',    values['p0'], values),
      'alpha' : lambda values: self.weighted('massFlux', values['alpha'], values),
      'T0'    : lambda values: values['T0'][0],
      'p01'   : lambda values: values['p01'][0],
      'p1'    : lambda values: values['p1'][0]
    }

    dicti['momentum'] = {
      'p'     : lambda values: self.weighted('area', values['p'], values),
      'p0'    : lambda values: self.weighted('area', values['p'], values)*(1-(self.fluid['gamma']-1)*self.weighted('massFlux', self.measuredVariables['v_mag'](values), values)**2/(2*self.fluid['gamma']*self.fluid['r']*values['T0'][0]))**(self.fluid['gamma']/(1-self.fluid['gamma'])),
      'alpha' : lambda values: self.angleAsArctanOfFluxes(values),
      'T0'    : lambda values: values['T0'][0],
      'p01'   : lambda values: values['p01'][0],
      'p1'    : lambda values: values['p1'][0]
    }

    dicti['enthalpy'] = {
      'p'     : lambda values: self.weighted('area',     values['p'], values),
      'p0'    : lambda values: self.weighted('area',     values['p'], values)*((self.weighted('massFlux', self.measuredVariables['T'](values), values))/values['T0'][0])**(self.fluid['gamma']/(1-self.fluid['gamma'])),
      'alpha' : lambda values: self.angleAsArctanOfFluxes(values),
      'T0'    : lambda values: values['T0'][0],
      'p01'   : lambda values: values['p01'][0],
      'p1'    : lambda values: values['p1'][0]
    }

    dicti['entropy'] = {
      'p'     : lambda values: self.weighted('area',     values['p'], values),
      'p0'    : lambda values: np.exp(self.weighted('massFlux', np.log(values['p0']), values)),
      'alpha' : lambda values: self.angleAsArctanOfFluxes(values),
      'T0'    : lambda values: values['T0'][0],
      'p01'   : lambda values: values['p01'][0],
      'p1'    : lambda values: values['p1'][0]
    }

    return dicti

  @reductionMethod
  def reduction_universalAveraging(self, kind : str ):
    reduced = { variable : self.averagingFunctions[kind][variable](self.rawData) for variable in self.averagingFunctions[kind].keys()}
    reduced['uncertainties'] = self.computeUncertainties(self.averagingFunctions[kind], reduced, self.rawData)
    
    reduced.update(self.textInfoAboutAveraging(kind))
    reduced['fluxes'], reduced['fluxesUncertainties'] = self.meanFluxes(reduced)
    reduced = self.otherVariables_fromMeasured(reduced, reduced['uncertainties'])
    return reduced

  def computeUncertainties(self, dictionaryOfLambdas : dict, dictionaryOfResults : dict, dictionaryOfData : dict):
    res = dict()
    for variable in dictionaryOfLambdas.keys(): 
      try: 
        size = len(dictionaryOfResults['p'])
      except: 
        try:
          size = len(dictionaryOfResults['I_M'])
        except: 
          size = 1
      localUncs = np.empty((len(self.givenUncertainties), size))
      
      for index,variable2 in enumerate(self.givenUncertainties.keys()):
        localDicti = { v : dictionaryOfData[v] for v in dictionaryOfData.keys()}
        localDicti[variable2] = dictionaryOfData[variable2] + self.givenUncertainties[variable2]

        localUncs[index, :] = dictionaryOfLambdas[variable](localDicti) - dictionaryOfResults[variable]
      res[variable] = np.linalg.norm(localUncs, axis=0)
      if len(res[variable]) == 1: res[variable] = res[variable][0]

    return res

if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')