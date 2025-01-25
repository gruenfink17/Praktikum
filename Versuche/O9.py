import numpy as np
from uncertainties import ufloat
import uncertainties.umath as umath
import uncertainties.unumpy as unp
from functions import *
T=ufloat(15.7e-3, 0.2e-3)#s
delta_t=ufloat(700e-6,100e-6)#s
L_0=ufloat(2.2e-3,0.2e-3)#m
lambda_0=633e-9#m
Finesse=T/delta_t
print("Finesse=",Finesse)

R=(np.pi**2+2*Finesse**2+np.pi*umath.sqrt(np.pi**2+4*Finesse**2))/(2*Finesse**2)
R_2=(np.pi**2+2*Finesse**2-np.pi*umath.sqrt(np.pi**2+4*Finesse**2))/(2*Finesse**2)
print("R=",R)
print("R_2=",R_2)

#teil 1.2
T_2=ufloat(15.4e-3,0.4e-3)#s
Delta_t=ufloat(2.1e-3,0.3e-3)#s
Delta_lambda=(lambda_0**2*Delta_t)/(2*L_0*T_2)
print("Delta_lambda=",Delta_lambda)


#Teil 2
ordnung=np.arange(1,13,1)#Ordnung i des Maximums
Pixelbreite=4.8e-6#m
#mit Unsicherheiten
#Position R1_unten (in px)
uR_1_u=ufloat(525,2)
uR_1_o=ufloat(335,2)
#Mittelpunkt
uAbstand_1=(uR_1_u-uR_1_o)
uM=uR_1_o+(uAbstand_1/2)
print("Mittelpunkt,", uM)
#Positionen von innen nach außen
uPositionen=unp.uarray([335,293,261,234,208,186,165,146,126,109,91,76],[2,2,2,2,2,2,2,2,2,2,2,2])
#Radius
uRadien=uM-uPositionen
#Durchmesser in Pixel
uD_p=uRadien*2
#Durchmesser in m
uD=uD_p*Pixelbreite#m
uf=ufloat(40e-3,0.4e-3)#m
ubeta=unp.arctan(uD/(2*uf))
print("uD für Fehler",uD)
uL=ufloat(3.95e-3,0.05e-3)
uwellenlängen_list=[0]
for i in range (len(ubeta)-1):
    uwellenlängen_list.append(uL*2*(unp.cos(ubeta[i])-unp.cos(ubeta[i+1])))

#uwellenlängen=unp.uarray(uwellenlängen_list)
print("Wellenlängen mit Fehler:", uwellenlängen_list)


#ohne Unsicherheiten (für Latex)
#Position R1_unten (in px)
R_1_u=525
R_1_o=335
#Mittelpunkt
Abstand_1=(R_1_u-R_1_o)
M=R_1_o+(Abstand_1/2)
#Positionen von innen nach außen
Positionen=np.array([335,293,261,234,208,186,165,146,126,109,91,76])
#Radius
Radien=M-Positionen
#Durchmesser in Pixel
D_p=Radien*2
#Durchmesser in m
Pixelbreite=4.8e-6#m
D=D_p*Pixelbreite#m
f=40e-3#m
beta=np.arctan(D/(2*f))

#wellenlängen berechnen
L=3.95e-3
wellenlängen_list=[0]
for i in range (len(beta)-1):
    wellenlängen_list.append(L*2*(np.cos(beta[i])-np.cos(beta[i+1])))

wellenlängen=np.array(wellenlängen_list)
#print(len(ordnung),len(Positionen), len(D), len(ubeta), len(wellenlängen))
print(latex([ordnung,Positionen,D*1000,ubeta,wellenlängen*10**9],["Ordnung","Position[px]","Di[mm]","beta","wellenlänge"]))
print("wellenlänge", mean(wellenlängen), "+/-",get_s_x(wellenlängen))


