## SPLEEN

```python
rename = {'y/g [-]'     :'x',
          'd [deg]'     :'alpha_2',
          'rho [kg/m^3]':'rho',
          'V_ax [m/s]'  :'v_ax',
          'P06/P01 [-]' :'p0/p01',
          'Ps6/P01 [-]' :'p/p01'},

add = { 'theorOutletAngle':np.deg2rad(53.8),
        'pitch'           :32.950e-3}
```
## KOBRA

```python
rename= {' a2 (°)'   :'alpha_2',
         ' T (K)'    :'T',
         ' p2 (kPa)' :'p',
         ' p02 (kPa)':'p0',
         ' z (1)'    :'loss',
         'y/t.1'     :'x'},
add={'scalePressure':1e3}
```

## SCHREIBER84

```python
add={'T0':303.15, 'pitch':0.031480, 'p01':1e5}
```
