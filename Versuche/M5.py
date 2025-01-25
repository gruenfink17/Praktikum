#---Teil 1---
#Größe der Amplitudenmodulation im Vergleich zum Gesamtsignal in den jeweiligen Ordnungen
import numpy as np
from functions import *


#---Teil 2---
#Frequenzposition von 5 aufeinanderfolgenden Maxima (m=5) des Beugungssignals
m=np.array([0,1,2,3,4])
f=np.array([4.3405, 4.3494, 4.3548,4.3587, 4.3664])#MHz

#h ermitteln
h=8.3#cm

#Delta f
df_Mhz=f-4.3405#Mhz
df=df_Mhz*1000#kHz

print(latex([m,f,df], ["m", "f [Mhz]", "Delta f [kHz]"]))
#lineare Regression deltaf(m)
print(wert_xy(m,df))
graph(m,df,trendlinie=True, title=r"lineare Regression $\Delta f(m)$", xlabel="m", ylabel=r"$\Delta f(m)$ in kHz", xlim=[0,5], ylim=[0,28])



#print(owert_xy(m,df))
#ugraph(m,df,trendlinie=True, title=r"lineare Regression $\Delta f(m)$", xlabel="m", ylabel=r"$\Delta f(m)$ in kHz", xlim=[0,5], ylim=[0,28])
#---Teil 3---
#Abstände der Beugungsordnungen pm1,2, ggf 3 als Funktion der Schallfrequenzen
#1. Ordnung
f_1=np.array([4,5,6,7,8,9]) #MHz
s_1=np.array([0.6,0.7,0.8,0.9,1.3,1.4])#cm
d_1=s_1/2 #cm
f_Hz=f_1*1000000
d_1_m=d_1/100
#print(latex([f_1,s_1,d_1],["f [MHz]", "s_1 [cm]", "d_1 [cm]"]))

#lineare Regression
print(wert_xy(f_Hz,d_1_m))
graph(f_1,d_1,True,r"Lineare Regression von $d_1(f)$", xlabel="f in MHz", ylabel=r"$d_1$ in cm", ylim=[0.0,0.8], xlim=[3,10])




#2.Ordnung
f_2=np.array([4,5,6,7,8,9])
s_2=np.array([1.3,1.3,1.5,1.7,1.9,2.0])
d_2=s_2/2
#print(latex([f_2,s_2,d_2],["f [MHz]", "s_2 [cm]", "d_2 [cm]"]))
d_2_m=d_2/100
#lineare Regression

print(owert_xy(f_Hz, d_2_m, ))
ograph(f_2, d_2, True, r"Lineare Regression von $d_2(f)$", xlabel="f in MHz", ylabel=r"$d_2$ in cm", ylim=[0.0, 1.1], xlim=[3, 10])