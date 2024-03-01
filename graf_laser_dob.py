# -*- coding: utf-8 -*-
"""
Created on Mon May  8 18:40:37 2023

@author: Juan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

dat= pd.read_excel(r"datos_l.xlsx")
datos=pd.DataFrame(dat,columns=["V4 (mV)", "X4 (mu m)"])
#print(datos) #335
ydata=(datos["V4 (mV)"][:292]).astype(float)
xdata=datos["X4 (mu m)"][:292].astype(float)
u=(xdata)-3660#2990
#print(u)
a=0.1
d=0.356
def sinc(x,I,A,B,C):
    return I*np.cos(B*x)**2*((np.sin(A*x+C))/(A*x+C))**2

guess=[140,np.pi/360,np.pi/360,1]
parameters, covariance = curve_fit(sinc,u, ydata,p0=guess)

fit_I,fit_A,fit_B,fit_C = parameters[0],parameters[1],parameters[2],parameters[3]
x=np.linspace(0,6640,500)

fig,ax=plt.subplots(2,1, figsize=(6,8))


ax[0].scatter(u,ydata)
ax[0].plot(u,sinc(u,fit_I,fit_A,fit_B,fit_C), linestyle="--",color="k",label="Ajuste por \n Fraunhofer")
ax[0].set_xlabel("distancia ($\mu$m)")
ax[0].set_ylabel("Voltaje (mV)")
ax[0].grid()
ax[0].set_title("Voltaje contra distancia en la pantalla")
ax[0].legend(loc="upper left")
print("parametros (I,A,B,C)",parameters)
print("incertidumbre (I,A,B,C)",covariance[0][0]**(1/2),covariance[1][1]**(1/2),covariance[2][2]**(1/2),covariance[3][3]**(1/2),)
print("longitud de onda (A)",np.pi*a/parameters[1],np.sqrt(np.pi**2*a**2/(parameters[1]**4)*covariance[1][1]))
print("longitud de onda (B)",np.pi*d/parameters[2],np.sqrt(np.pi**2*a**2/(parameters[2]**4)*covariance[2][2]))
#print(xdata)

res=ydata-sinc(u,fit_I,fit_A,fit_B,fit_C)



ax[1].scatter(u,res,c="r")
ax[1].grid()
ax[1].set_xlabel("distancia ($\mu$m)")
ax[1].set_ylabel("Residuales $R_i$ (mV)")
ax[1].set_title("Gráfica de residuales")
ax[1].legend()

plt.tight_layout()
plt.show()