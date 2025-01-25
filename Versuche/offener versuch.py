import numpy as np
from uncertainties import ufloat

from functions import *
import uncertainties as uc

#plus
n=np.array([-11,-10,-9,-8,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11])
n_calc=np.array([-11,-10,-9,-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11])
Hauptmaximum=mean([42.8,42.7])
s_1=np.array([43.8,44.6,45.2,46.0,47.9,48.7,49.6,50.6,51.8,52.7])
s_2=np.array([43.7,44.5,45.1,45.9,46.7,48.7,49.6,50.6,51.7,52.6])


plus_mean_list=[]
for i in range(len(s_1)):
    plus_mean_list.append(mean([s_1[i],s_2[i]]))
plus_mean=np.array(plus_mean_list)

#minus
s_1_minus=np.array([42.0,41.3,40.6,40.0,39.4,38.4,37.7,37.1,36.5,35.5,35.0])
s_2_minus=np.array([42.8,41.6,40.9,40.2,39.2,38.2,37.6,37.1,36.6,35.6,35.0])
minus_mean_list=[]
for i in range(len(s_1_minus)):
    minus_mean_list.append(mean([s_1_minus[i],s_2_minus[i]]))
minus_mean=np.array(minus_mean_list)

#ok umständlich sorry
minus_mean = minus_mean[::-1]
maxima_list=minus_mean.tolist()+plus_mean.tolist()
maxima=np.array(maxima_list)

#Abstände
abstände=np.abs(maxima-Hauptmaximum)
print(abstände)



#Wellenlänge/d
L=402e-3
d=ufloat(4.45e-3,0.05e-3)
ld=np.abs(1/((n_calc+0.5)*(np.sqrt(1+(L**2)/(abstände*0.01)**2))))
ld_mean=mean(ld)
Fehler_ld=get_s_x(ld)
print("Fehler",Fehler_ld)
ld_ufloat=ufloat(ld_mean,Fehler_ld)
print(ld_ufloat)
l=ld_ufloat*d
print("l",l)

print(latex([n_calc,abstände],["Ordnung", "s"]))








#find_lambda(L,d,abstände,n_calc)
