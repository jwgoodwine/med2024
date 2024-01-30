import numpy as np
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
import numfracpy as nfr

#plt.rcParams['text.usetex']=True
gain = np.random.uniform(5,9)
alpha = np.random.uniform(1.01,2)
def f(t,u):
  return (1 - u)*gain
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
print(alpha)
thissoln = np.zeros(shape=(1,101))
NumSol = nfr.FODE(f,[0,0],[0,10],0.01,alpha)
width = 3.5
height = width*(5**.5 - 1)/2
height = 1.1*height
print(height)
fig = plt.figure(figsize=(width,height))
plt.plot(NumSol[0],NumSol[1],label=r'$\alpha='+str(round(alpha,2))+'$',linewidth=1.5)
plt.xlabel("$t$")
plt.ylabel("speed")
plt.tight_layout()
print(f"saving plot")
plt.savefig("testfig.pgf",format="pgf")
print(f"done saving plot")

