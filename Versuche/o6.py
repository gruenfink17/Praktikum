import numpy as np
from functions import *
import uncertainties as uc


#O6, nicht A2

#wie achim uns das gesagt hat:
dn=uc.ufloat(13.25,0.25)
lamda=510e-9
dl_a=dn*lamda
print(dl_a)

#wie es im hinweisheft steht:
c=300e6#m/s
dt=17.8e-3
dl_b=c*dt
print(dl_b)


c=299792458#m/s
dt_ac=dl_a/c
print("dt:",dt_ac)

dlambda=lamda**2/(c*dt_ac)
print("dlambda:",dlambda)
