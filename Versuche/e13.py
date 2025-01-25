import numpy as np
from functions import *
from uncertainties import ufloat
from uncertainties import unumpy as unp
# E13

R_C = 5.6/(2*20*10**(-3))
print("R_C =", R_C)
R_E = -140/5
print("R_E =", R_E)
#R_B = 150* (5.6-0.7-20*10*(-3)(-28))/(20*10**(-3))
#print("R_B = ", R_B)
I_b = (20*10**(-3))/150
print("I_b =", I_b)
R_e = (0.026/I_b) + 150*(-28)
print("R_e =", R_e)
C_L = 1/(2*np.pi *3000 * R_e)
print("C_L =" , C_L)

U_a = np.array([344, 340, 369, 396, 451, 430, 187,235,270])
U_e = np.array([80, 77, 77, 77 ,79, 78, 78,78,79])
nv= U_e/U_a
R_E = np.array([31.27, 30.33, 28.04, 26.67, 22.96, 24.51, 63,49.0,42.31])
#print(latex([R_E,U_e,U_a,nv],["RE","Ue","Ua","1nu"]))
#ograph(R_E, nv,True,xlabel=r"$R_E[\Omega]$",ylabel=r"$1/\nu$",title=r"$1/\nu(R_E)$")
print(wert_xy(R_E,nv))
print(owert_xy(R_E, nv))
nRCsoll=1/R_C
print ("nRCsoll =", nRCsoll)

#nochmal kurz für Unsicherheiten
uU_a=unp.uarray([344, 340, 369, 396, 451, 430, 187,235,270], [5,5,5,5,5,5,5,5,5])
uU_e=unp.uarray([80, 77, 77, 77 ,79, 78, 78,78,79],[5,5,5,5,5,5,5,5,5])
unv=uU_e/uU_a
#print(unv)

 #Frequenzabhängigkeit
f=np.array([600,1000,5000,10e3,50e3,100e3,300e3,500e3,1e6,2e6])#Hz
U_ein=np.array([78,77,77,77,77,77,77,77,77,73])#mV
U_aus=np.array([81,135,330,355,370,380,360,360,365,375])#mV
nu=U_aus/U_ein
print(latex([f,U_ein,U_aus,nu],["f","U_e","U_aus","nu"]))
graph(f,nu,xlabel="f[Hz]",ylabel=r"$\nu$",xlog=True)

uU_ein=ufloat(78,1)
uU_aus=ufloat(81,5)
unu=uU_aus/uU_ein
print(unu)

 #Multivibrator
