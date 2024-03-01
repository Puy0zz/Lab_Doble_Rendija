# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:37:33 2024

@author: Adriana
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

dat= pd.read_excel(r"datos_l.xlsx")
datos=pd.DataFrame(dat,columns=["V3 (mV)", "X3 (mu m)"])
#print(datos) #184
ydata=(datos["V3 (mV)"][80:177]).astype(float)
xdata=datos["X3 (mu m)"][80:177].astype(float)
u=(xdata)-5640#2990

a=0.1
d=0.356
#print(u)
"""
def sinc(x,I,B,C):
    return I*np.cos(B*x+C)**2
"""
def sinc(x,I,B,C):
    return I*((np.sin(B*x+C))/(B*x+C))**2



guess=[300,np.pi/670,10]
parameters, covariance = curve_fit(sinc,u, ydata,p0=guess)

fit_I,fit_B,fit_C = parameters[0],parameters[1],parameters[2]
x=np.linspace(-3000,1000,1000)

fig, ax= plt.subplots(2,1, figsize=(6,8))


ax[0].scatter(u,ydata)
ax[0].plot(u,sinc(u,fit_I,fit_B,fit_C), linestyle="--",color="k",label="Ajuste por \n Fraunhofer")
ax[0].set_xlabel("distancia ($\mu$m)")
ax[0].set_ylabel("Voltaje (mV)")
ax[0].grid()
ax[0].set_title("Voltaje contra distancia en la pantalla")
ax[0].legend(loc="upper left")
print("parametros (I,B,C)",parameters)
print("incertidumbre (I,B,C)",covariance[0][0]**(1/2),covariance[1][1]**(1/2),covariance[2][2]**(1/2))
print("longitud de onda",np.pi*a/parameters[1],np.sqrt(np.pi**2*a**2/(parameters[1]**4)*covariance[1][1]))
#print(xdata)


res=ydata-sinc(u,fit_I,fit_B,fit_C)



ax[1].scatter(u,res,c="r")
ax[1].grid()
ax[1].set_xlabel("distancia ($\mu$m)")
ax[1].set_ylabel("Residuales $R_i$ (mV)")
ax[1].set_title("Gráfica de residuales")
ax[1].legend()
plt.tight_layout()
plt.show()