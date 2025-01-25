import numpy as np
from functions import *
import uncertainties as uc

e=1.602177e-19
print(e)
#-----Neon------
#1a
Ordnung_1a= np.array([3,4])
Ordnung_tab=np.array([3,4,3,4,3,4,3,4])
dU1a1=np.array([41.35-23.25,60.35-41.35 ])
dU1a2=np.array([41.80-24.10, 62.10-41.80])
dU1a3=np.array([42.10-24.90, 62.95-42.10])
dU1a4=np.array([43.35-25.25,63.80-43.35])
dU1a=np.array([41.35-23.25,60.35-41.35, 41.80-24.10, 62.10-41.80,42.10-24.90, 62.95-42.10, 43.35-25.25,63.80-43.35])
dE1a=dU1a*e


#print(latex([Ordnung_tab,dU1a,dE1a], ["Ordnung", "Delta UB", "Delta E"]))
#print(wert_xy(Ordnung_tab,dU1a, "lreg"))
#graph(Ordnung_tab,dU1a,True)

#1b
Ordnung_1b=np.array([3,3,4,3,4,3,4])
dU1b=np.array([40.85-23.00, 42.20-23.85, 61.15-42.20,42.75-24.65, 62.05-42.75, 43.60-25.50, 63.15-43.60])#hier hab ich das beim letzten geändert weil das bei excel auch ganze anders aussieht
dE1b=dU1b*e
#print(latex([Ordnung_1b, dU1b], ["Ordnung", "Delta UB"]))

#2a
Ordnung_2a=np.array([3,4,3,3,3])
dU2a=np.array([41.55-23.50,61.50-41.55,41.55-23.20,40.70-22.65,40.20-22.40])
dE2a=dU2a*e
#print(latex([Ordnung_2a,dU2a], ["Ordnung", "Delta UB"]))

#2b
Ordnung_2b=np.array([3,4,3,4,3,3])
dU2b=np.array([44.00-24.95,63.85-44.00, 43.55-24.95,63.30-43.55,43.05-24.45,42.70-24.10])
dE2b=dU2b*e
#print(latex([Ordnung_2b, dU2b], ["Ordnung", "Delta UB"]))

#lineare Regression
#alle Sachen als Listen zum zusammenfügen
list_O1a=Ordnung_tab.tolist()
list_O1b=Ordnung_1b.tolist()
list_O2a=Ordnung_2a.tolist()
list_O2b=Ordnung_2b.tolist()
Ordnungen_list=list_O1a+list_O1b+list_O2a+list_O2b

list_dU1a=dU1a.tolist()
list_dU1b=dU1b.tolist()
list_dU2a=dU2a.tolist()
list_dU2b=dU2b.tolist()
dU_list=list_dU1a+list_dU1b+list_dU2a+list_dU2b

#wieder zu arrays
Ordnungen=np.array(Ordnungen_list)
dU=np.array(dU_list)

#print(wert_xy((Ordnungen-1),dU))
#graph(Ordnungen-1,dU,True, r"Lineare Regression $\Delta U_B(n)$ ", xlabel="n", ylabel=r"$\Delta U_B$ [V]", xlim=[0,5], ylim=[15,21])

#Quecksilber
#a
Ordnung1=np.array([4,5,6,7])
Q41=mean([4.69,4.75,4.80,4.80,4.80])
Q51=mean([4.69,4.92,4.84,4.94,4.92])
Q61=mean([4.99,5.11,5.11,5.08,5.01])
Q71=mean([4.99,4.84,4.97,5.01])
dUQ1=np.array([Q41,Q51,Q61,Q71])
Stabw41=get_s_x([4.69,4.75,4.80,4.80,4.80])
Stabw51=get_s_x([4.69,4.92,4.84,4.94,4.92])
Stabw61=get_s_x([4.99,5.11,5.11,5.08,5.01])
Stabw71=get_s_x([4.99,4.84,4.97,5.01])
print("Stabw:", Stabw41, Stabw51, Stabw61, Stabw71)

M11=np.array([4.69,4.69,4.99,0])
M21=np.array([4.75,4.92,5.11,4.99])
M31=np.array([4.80,4.84,5.11,4.84])
M41=np.array([4.80,4.94,5.08,4.87])
M51=np.array([4.80,4.92,5.01,5.01])
print(latex([Ordnung1,M11,M21,M31,M41,M51,dUQ1], ["Ordnung","Messung 1", "Messung 2", "Messung 3", "Messung 4", "Messung 5", "Mittelwert"]))
print(wert_xy(Ordnung1, dUQ1))
graph(Ordnung1,dUQ1, True, title=r"Lineare Regression von $\Delta U_B(n)$ für $T=155^\circ C$", xlabel= "n", ylabel=r"$\Delta U_B$", xlim=[3,13])

#b
Ordnung2=np.array([6,7,8,9,10,11,12])
M12=np.array([4.80,4.88,4.84,4.80,4.84,5.04,5.00])
M22=np.array([4.88,4.84,4.84,4.72,4.96,4.96,4.96])
M32=np.array([4.88,4.68,4.80,4.96,4.88,4.92,5.04])
M42=np.array([4.76,4.72,4.96,4.76,4.88,5.00,5.04])
M52=np.array([4.88,4.72,4.92,4.76,5.04,4.88,4.88])

mylist=[]
for i in range(len (M12)):
    mylist.append(mean([M12[i],M22[i],M32[i],M42[i],M52[i]]))
dUQ2=np.array(mylist)


print(latex([Ordnung2,M12,M22,M32,M42,M52, dUQ2], ["Ordnung","Messung 1", "Messung 2", "Messung 3", "Messung 4", "Messung 5", "Mittelwert"]))
print(wert_xy(Ordnung2, dUQ2))
#graph(Ordnung2,dUQ2, True, title=r"Lineare Regression von $\Delta U_B(n)$ für $T=190^\circ C$", xlabel= "n", ylabel=r"$\Delta U_B$", xlim=[5,13])


L=uc.ufloat(2.3e-5,0.8e-5)
sigma=1/(2.623e23*L)
print(sigma)

