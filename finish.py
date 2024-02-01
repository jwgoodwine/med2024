NN = 500
predictions = np.zeros(shape=(NN,2))
fig = plt.figure(figsize=(8,5))

for n in range(NN):
  print(n)
  gain = np.random.uniform(5,9)
  alpha = np.random.uniform(0.75,2.1)
  if alpha <= 1:
    Initial = [0]
  elif alpha > 2:
    Initial = [0,0,0]
  else:
    Initial = [0,0]
  
  def f(t,u):
    return (1 - u)*gain

  print(alpha)
  thissoln = np.zeros(shape=(1,101))
  NumSol = nfr.FODE(f,Initial,Interv,dx,alpha)
  thissoln[0] = np.interp(t,NumSol[0],NumSol[1])
  plt.plot(NumSol[0], NumSol[1], label=r'$\alpha='+str(round(alpha,2))+'$',linewidth=1.5)
  thissoln=scaler.transform(thissoln)
  thissoln = torch.from_numpy(thissoln).to(torch.float32)
  out = model.forward(thissoln).detach().numpy()
  out = out[0,0]
  print(out)
  predictions[n,:] = [alpha,out]

plt.legend(loc=(1,0.2),shadow=False,ncol=1,prop={'size':16},frameon=False)
plt.grid(True)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel(r'$t$',size=23)
plt.ylabel(r'$u(t)$',size=23)
plt.show()

width = 3.5
height = width*(5**.5-1)/2
height = 1.1*height
fit = plt.figure(figsize=(width,height))

plt.scatter(predictions[:,0],predictions[:,1],marker=".")
plt.grid(True)
regr = linear_model.LinearRegression()
regr.fit(predictions[:,0].reshape(-1,1),predictions[:,1].reshape(-1,1))
linpredict = regr.predict(predictions[:,0].reshape(-1,1))
print(mean_squared_error(predictions[:,1].reshape(-1,1),linpredict))
print(r2_score(predictions[:,1].reshape(-1,1),linpredict))
plt.plot(predictions[:,0],linpredict,color="k")
plt.xlabel("Input order, $\\alpha$")
plt.ylabel("Predicted order")
plt.tight_layout()
plt.savefig("predicted.pgf",format="pgf")

