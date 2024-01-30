
Initial = [0,0]
Interv = [0,tfinish]
dx = 0.01

NN = 1000
predictions = np.zeros(shape=(NN,2))
fig = plt.figure(figsize=(8,5))

for n in range(NN):
  print(n)
  gain = np.random.uniform(4,10)
  alpha = np.random.uniform(1.01,2)
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

fig2 = plt.figure(figsize=(8,5))
plt.scatter(predictions[:,0],predictions[:,1],marker=".")
plt.show()


