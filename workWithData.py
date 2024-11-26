import numpy as np
import pandas as pd
import auxiliaryFunctions as aux

class TraversingData:
  """
  Class for representation and manipulation with traversing data.
  """

  def __init__(self, data:pd.DataFrame, ready=False, rename:dict=dict(), add:dict=dict(),
               turbine=True, fluid = {'r':8314.3/28.96, 'gamma':1.4}):
    self.fluid = fluid
    if 'cp' not in self.fluid.keys(): self.fluid['cp'] = self.fluid['gamma']*self.fluid['r']/(self.fluid['gamma']-1)

    self.rawData = data
    self.add = add
    self.turbine = turbine; self.compressor = (not turbine)
    
    if not ready: self.prepro(rename)
    if ready: self.t01 = add['t01']
    self.p01 = np.mean(self.rawData['p01'])
    if not self.turbine: self.p1 = add['p1']

    self.dataPitch = self.rawData['x'].max()-self.rawData['x'].min()
    if not (np.isclose(self.dataPitch,1, rtol=0.05)
            or 
            np.isclose(self.dataPitch,add['pitch'], rtol=0.05)):
      print(f'The data might not reference one pitch. The data pitch is {self.dataPitch:.3f}')
    
  def prepro(self,rename):
    """Preprocess traversing data."""

    self.rawData.rename(columns=rename, 
                       inplace=True)
    
    if 'scalePressure' in  self.add.keys():
      self.rawData['p']  *=  self.add['scalePressure']
      self.rawData['p0'] *=  self.add['scalePressure']

    if 'alpha_2' in self.rawData.keys():
      self.rawData['alpha_2'] = np.deg2rad(self.rawData['alpha_2'])

    if 'theorOutletAngle' in  self.add.keys(): 
      self.rawData['alpha_2'] = self.rawData['alpha_2'] +  self.add['theorOutletAngle']

    if 'p/p01' in self.rawData.keys():
      self.rawData['p/p0'] = self.rawData['p/p01']/self.rawData['p0/p01']
      self.rawData['Ma'] = aux.ma_is(self.rawData['p/p0'], 1, self.fluid['gamma'])

      self.rawData['T'] = (self.rawData['v_ax']/np.cos(self.rawData['alpha_2'])/self.rawData['Ma'])**2/self.fluid['gamma']/self.fluid['r']
      
      a_t0 = self.rawData['T']+(self.rawData['v_ax']/np.cos(self.rawData['alpha_2']))**2/2/self.fluid['cp']

      self.rawData['p']  = self.fluid['r']*self.rawData['rho']*self.rawData['T']
      self.rawData['p0'] = self.rawData['p']/self.rawData['p/p0']
      self.rawData['p01']= self.rawData['p']/self.rawData['p/p01']
      self.rawData['v_tan'] = self.rawData['v_ax']*np.tan(self.rawData['alpha_2'])

    elif 'Points:0' in self.rawData.keys():
      a_t0 =  self.add['T0']*np.ones(len(self.rawData.index))

      self.rawData.sort_values('Points:0', inplace=True)
      self.rawData['x'] = np.sqrt(self.rawData['Points:0']**2 + self.rawData['Points:1']**2 )  
      self.rawData['x'] -= self.rawData['x'].min()

      alpha_2 = np.empty(len(self.rawData.index))

      firstPoint = np.array([self.rawData['Points:0'][0], self.rawData['Points:1'][0]])
      lastPoint  = np.array([self.rawData['Points:0'][len(self.rawData.index)-1], self.rawData['Points:1'][len(self.rawData.index)-1]])
      tangent = firstPoint-lastPoint; tangent /= np.linalg.norm(tangent)
      outletNorm = np.array([-tangent[1], tangent[0]])

      for i in range(len(self.rawData.index)):
        velVec = np.array([self.rawData['U:0'][i], self.rawData['U:1'][i]]); velVec/=np.linalg.norm(velVec)
        alpha_2[i] = np.arccos(np.dot(outletNorm, velVec))
      
      self.rawData['alpha_2'] = alpha_2

      self.rawData['rho'] = self.rawData['p']/self.rawData['T']/self.fluid['r']
      umag = np.sqrt(self.rawData['U:0']**2 + self.rawData['U:1']**2)
      self.rawData['Ma'] = umag /np.sqrt( (self.fluid['gamma']*self.fluid['r']*self.rawData['T']) )

      self.rawData['p0'] = self.rawData['p']/aux.p(self.rawData['Ma'], 1, self.fluid['gamma'])
      self.rawData['p/p0'] = self.rawData['p']/self.rawData['p0']
      self.rawData['v_ax']  = umag*np.cos(self.rawData['alpha_2'])
      self.rawData['v_tan'] = umag*np.sin(self.rawData['alpha_2'])

      self.rawData['p01']  = np.ones(len(self.rawData.index))* self.add['p01']
    
    else:
      self.rawData['p/p0'] = self.rawData['p']/self.rawData['p0']
      a = (self.fluid['gamma']-1)/self.fluid['gamma']

      self.rawData['p01'] = self.rawData['p'] * (1-(1-self.rawData['p/p0']**a)/(1-self.rawData['loss']))**(-1/a)
      
      self.rawData['Ma'] = aux.ma_is(self.rawData['p/p0'], 1, self.fluid['gamma'])
      t_over_to  = aux.t(self.rawData['Ma'], gamma=self.fluid['gamma'])
      a_t0 = np.mean(self.rawData['T']/t_over_to)

      self.rawData['rho']  = self.rawData['p']/(self.fluid['r']*self.rawData['T'])

      self.rawData['v_ax']  = self.rawData['Ma']*np.sqrt(self.fluid['gamma']*self.fluid['r']*self.rawData['T'])*np.cos(self.rawData['alpha_2'])
      self.rawData['v_tan'] = self.rawData['v_ax']*np.tan(self.rawData['alpha_2'])
    
    self.t0 = np.mean(a_t0)
    self.t0_std = np.std(a_t0)

  def fluxes(self):
    if not (hasattr(self, 'trueFluxes')):
      print('Computing fluxes.')
      self.trueFluxes = aux.fluxesInt(self.rawData, np.mean(self.rawData['p01']),
                                      self.dataPitch, self.fluid)

    return self.trueFluxes
  
  def reduction_amecke95(self, normal = True):

    reduced = dict()
    fluxes = self.fluxes()

    d = fluxes['i_f']**2-4*(1-self.fluid['r']/2/self.fluid['cp'])*(self.t0*self.fluid['r']*fluxes['i_m']**2 - self.fluid['r']/2/self.fluid['cp'] * fluxes['i_c']**2)
    z = (fluxes['i_f']+(-1)**normal*np.sqrt(d))/2/(1-self.fluid['r']/2/self.fluid['cp'])

    reduced = {
      'p'    : fluxes['i_f']-z,
      'rho'  : fluxes['i_m']**2/z,
      'v_tan': fluxes['i_c']/fluxes['i_m']
    }

    reduced['T'] = reduced['p']/reduced['rho']/self.fluid['r']
    reduced['v_ax'] = fluxes['i_m']/reduced['rho']
    reduced.update({
      'alpha_2':np.arctan2(reduced['v_tan'], reduced['v_ax']),
      'Ma':np.sqrt( (2*self.fluid['cp']*(self.t0-reduced['T'])) / (self.fluid['gamma']*self.fluid['r']*reduced['T']) ),
    })
    p_over_p0 = aux.p(reduced['Ma'])
    reduced['p0'] = reduced['p']/p_over_p0

    if self.turbine:
      reduced['loss'] = aux.kineticEnergyLossCoefficient(reduced['p0'], self.p01, reduced['p'], self.fluid['gamma'])
    else:
      reduced['loss'] = aux.totalPressureLossCoefficient(reduced['p0'], self.p01, reduced['p'], self.p1, self.fluid['gamma'])

    reduced['fluxes'] = aux.fluxesDisc(reduced, 
                                       np.mean(self.rawData['p01']),
                                       self.fluid)
    return reduced
  
  def reduction_vzlu(self):
    fluxes = self.fluxes()

    reduced = {
      'p0':np.mean(self.rawData['p01'])*np.exp(-1/self.fluid['r']*fluxes['i_s']/fluxes['i_m']),
      't' :self.t0*fluxes['i_h']/fluxes['i_m'],
      'alpha_2':np.arctan2(fluxes['i_c'],fluxes['i_a']),
    }
    reduced['p']    = reduced['p0']*(fluxes['i_h']/fluxes['i_m'])**(self.fluid['gamma']/(self.fluid['gamma']-1))

    reduced['Ma']   = aux.ma_is(reduced['p'], reduced['p0'])
    reduced['v_ax']  = fluxes['i_a']/fluxes['i_m']
    reduced['v_tan'] = fluxes['i_c']/fluxes['i_m']
    reduced['rho']  =  fluxes['i_m']/reduced['v_ax']

    if self.turbine:
      reduced['loss'] = aux.kineticEnergyLossCoefficient(reduced['p0'], self.p01, reduced['p'], self.fluid['gamma'])
    else:
      reduced['loss'] = aux.totalPressureLossCoefficient(reduced['p0'], self.p01, reduced['p'], self.p1, self.fluid['gamma'])

    reduced['fluxes'] = aux.fluxesDisc(reduced, 
                                       np.mean(self.rawData['p01']),
                                       self.fluid)

    return reduced

  def reduction_massFlux(self):
    reduced = dict()

    arg1 = self.rawData['v_ax']*self.rawData['rho']
    m = np.trapz(arg1, self.rawData['x'])
    for v in ['p','p0','v_ax','v_tan']: 
      reduced[v] = np.trapz(self.rawData[v]*arg1, self.rawData['x'])/m
    
    reduced['Ma']  = aux.ma_is(reduced['p'], reduced['p0'], self.fluid['gamma'])
    reduced['rho'] = aux.rho(reduced['Ma'], reduced['p0']/self.fluid['r']/self.t0)
    reduced['T']   = aux.t(reduced['Ma'], self.t0) 
    reduced['alpha_2'] = np.arctan2(reduced['v_tan'], reduced['v_ax'])

    reduced['fluxes'] = aux.fluxesDisc(reduced, 
                                       np.mean(self.rawData['p01']),
                                       self.fluid) 

    if self.turbine:
      reduced['loss'] = aux.kineticEnergyLossCoefficient(reduced['p0'], self.p01, reduced['p'], self.fluid['gamma'])
    else:
      reduced['loss'] = aux.totalPressureLossCoefficient(reduced['p0'], self.p01, reduced['p'], self.p1, self.fluid['gamma'])

    return reduced

  def reduction_area(self):
    reduced = dict()

    for v in ['p','p0','v_ax','v_tan']:
      reduced[v] = np.trapz(self.rawData[v], self.rawData['x'])/self.dataPitch
    
    reduced['Ma']  = aux.ma_is(reduced['p'], reduced['p0'], self.fluid['gamma'])
    reduced['rho'] = aux.rho(reduced['Ma'], reduced['p0']/self.fluid['r']/self.t0)
    reduced['T']   = aux.t(reduced['Ma'], self.t0) 
    reduced['alpha_2'] = np.arctan2(reduced['v_tan'], reduced['v_ax'])

    if self.turbine:
      reduced['loss'] = aux.kineticEnergyLossCoefficient(reduced['p0'], self.p01, reduced['p'], self.fluid['gamma'])
    else:
      reduced['loss'] = aux.totalPressureLossCoefficient(reduced['p0'], self.p01, reduced['p'], self.p1, self.fluid['gamma'])

    reduced['fluxes'] = aux.fluxesDisc(reduced, np.mean(self.rawData['p01']), self.fluid) 

    return reduced

  def plot(self, amecke_normal=True):
    am = self.reduction_amecke95(amecke_normal)
    ar = self.reduction_area()
    ma = self.reduction_massFlux()
    vz = self.reduction_vzlu()

    methods        = [am,ar,ma,vz]
    names_methods  = ['MM', 'area', 'mass flux', 'VZLU']
    variables = ['alpha_2','Ma','loss']
    names_variables = [r'$\alpha_2 \approx$', r'$M_{2}\approx$',r'$\xi\approx$' if self.turbine else r'$\omega\approx$']
    fluxes = ['i_m','i_f','i_a','i_c','i_h','i_s']
    names_fluxes = [r'$I_M$', r'$I_A$', r'$I_C$',r'$I_H$', r'$I_S$']

    differences = dict()
    colors_differences = dict()
    for v in variables: 
      differences[v] = np.zeros((len(methods)-1, len(methods)-1))
      colors_differences[v] = np.zeros((len(methods)-1, len(methods)-1))

    for i in range(len(methods)-1):
      for j in range(i,len(methods)-1):
        for v in variables:
          differences[v][i][j] = methods[i][v]-methods[j+1][v]
          colors_differences[v][i][j] = methods[i][v]/methods[j+1][v]-1

    differences['alpha_2'] = np.rad2deg(differences['alpha_2'])
    
  
    fluxesErrors = np.zeros((len(methods), len(fluxes)))

    for i in range(len(methods)):
      method = methods[i]
      for j in range(len(fluxes)):
        nameFlux = fluxes[j]
        fluxesErrors[i,j] = (method['fluxes'][nameFlux]/self.fluxes()[nameFlux] -1)*100
    
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(ncols=4, figsize=(8.3, 1))

    for h in range(len(variables)):
      v = variables[h]
      ax[h].imshow(colors_differences[v], vmin=-1.1, vmax=1.1, cmap='bwr', aspect=1/3)
      for (j,i),label in np.ndenumerate(differences[v]):
        if i>=j:
          ax[h].text(i,j,f'{label :.3f}',ha='center',va='center', fontsize=8)

      ax[h].text(0,2,names_variables[h],ha='center',va='center')  
      valToShow = np.mean([m[v] for m in methods])
      if 'alpha' in names_variables[h]: valToShow = np.rad2deg(valToShow)
      ax[h].text(1,2,f'{valToShow:.3f}'+('alpha' in names_variables[h])*'°',ha='center',va='center')

    ax[-1].imshow(fluxesErrors, vmin=-10, vmax=10, cmap='bwr', aspect=1/(3*3/4))
    for (j,i),label in np.ndenumerate(fluxesErrors):
      ax[-1].text(i,j,f'{label :.3f}',ha='center',va='center', fontsize=8)

    ax[-1].tick_params(top=False,   labeltop=True, bottom=False, labelbottom=False,
                      right=False, labelright=True,   left=False, labelleft=False,)

    ax[-1].set_yticks(     range(len(names_methods)))
    ax[-1].set_xticks(     range(len(names_fluxes )))
    ax[-1].set_yticklabels(names_methods)
    ax[-1].set_xticklabels(names_fluxes )

    import matplotlib
    line = matplotlib.lines.Line2D((0.68,0.68),(0,1), color='k',transform=fig.transFigure)
    fig.lines = line,

    for ak in ax[:-1]:
      ak.set_xticks(     [])
      ak.set_yticks(     [])
      ak.set_xticklabels([])
      ak.set_yticklabels([])

    fig.subplots_adjust(
      top=0.9,
      bottom=-0.1,
      left=0.0,
      right=0.9,
      hspace=0.,
      wspace=0.1
    )
    
    return fig, ax





if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')