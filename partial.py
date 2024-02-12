
#matcontent = scipy.io.loadmat("ident.mat")
#ts = np.zeros(shape=(1,101))
#dat = matcontent['ident']
#dat = dat.transpose()
#ts[0] = dat[1]
#out = model.forward(torch.from_numpy(ts).to(torch.float32)).detach().numpy()


width = 3.5
height = width*(5**.5-1)/2
height = 1.1*height
fig = plt.figure(figsize=(width,height))
plt.grid(True)
plt.plot(dat.T[:,0],dat.T[:,1])
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.tight_layout()
plt.savefig("networkresponse.pgf",format="pgf")


