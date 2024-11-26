import numpy as np
import pandas as pd

from workWithData import TraversingData

itCas = 'my_data/trw1/M0.601.csv'
trw = TraversingData(pd.read_csv(itCas, delimiter=';', header=0),
               rename= {'a2'    :'alpha_2',
                       'p2'     :'p',
                       'p02'    :'p0',
                       'dzeta'  :'loss',
                       'y/t'    :'x'})
print(trw.fluxes())
print(trw.reduction_amecke95()['fluxes'])
print(trw.reduction_vzlu()['fluxes'])
print(trw.reduction_massFlux()['fluxes'])
print(trw.reduction_area()['fluxes'])

"""
kobra = 'my_data/Kobra_SS_Traversing_regime.xlsx'
TraversingData(pd.read_excel(kobra, sheet_name='61', skiprows=28, usecols='N:U'),
               rename= {' a2 (°)'   :'alpha_2',
                        ' T (K)'    :'T',
                        ' p2 (kPa)' :'p',
                        ' p02 (kPa)':'p0',
                        ' z (1)'    :'loss',
                        'y/t.1'     :'x'},
                add={'scalePressure':1e3})

regime='SPLEEN_C1_NC_St000_Re70_M070_PL06_L5HP_s5000'
spleen = f'/home/terezie/Downloads/SPLEEN_HighSpeedTurbineCascade_Database_v5/Experimental_DataBase/1_SPLEENC1_NC_WGOFF/PL06/L5HP/Area/{regime}.xlsx'
TraversingData(pd.read_excel(spleen),
               rename={'y/g [-]'     :'x',
                       'd [deg]'     :'alpha_2',
                       'rho [kg/m^3]':'rho',
                       'V_ax [m/s]'  :'v_ax',
                       'P06/P01 [-]' :'p0/p01',
                       'Ps6/P01 [-]' :'p/p01'},
               add = {'theorOutletAngle':np.deg2rad(53.8),
                      'pitch':32.950e-3})

schreiber84 = 'my_data/schreiber84/p0-1.0e5_Ma-0.95_B-147.0.csv'
TraversingData(pd.read_csv(schreiber84, delimiter=',', header=0),
               add={'T0':303.15, 'pitch':0.031480, 'p01':1e5})"""