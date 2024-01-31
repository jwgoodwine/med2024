import numpy as np
import control as ct
import numfracpy as nfr
import scipy.special as special
import matplotlib.pyplot as plt


Initial1 = [0]
Initial2 = [0,0]
Interv = [0,500]
dx = 0.1
gain = 5;

fig = plt.figure(figsize=(8,5))
def f(t,u):
    return (1 - u)/gain

#for alpha in [0.01, 0.25, 0.5, 0.75, 1.01, 1.25, 1.5, 1.75, 1.99]:
for alpha in [0.7]:
    print(alpha)
    if alpha < 1:
      NumSol = nfr.FODE(f,Initial1,Interv,dx,alpha)
    else:
      NumSol = nfr.FODE(f,Initial2,Interv,dx,alpha)
    plt.plot(NumSol[0], NumSol[1], label=r'$\alpha='+str(round(alpha,2))+'$',linewidth=1.5)

plt.legend(loc=(1,0.2),shadow=False,ncol=1,prop={'size':16},frameon=False)
plt.grid(True)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel(r'$t$',size=23)
plt.ylabel(r'$u(t)$',size=23)
plt.show()

