
# hand validation section
for n in range(5):
    thissoln = np.zeros(shape=(1,101))
    den = np.random.uniform(0.01, 4, 3)
    den[1] = 0
    t, yout = ct.step_response(ct.TransferFunction(den[2],den),t)
    thissoln[0] = yout
    thissoln=scaler.transform(thissoln)
    thissoln = torch.from_numpy(thissoln)
    thissoln = thissoln.to(torch.float32) 
    print(model.forward(thissoln))

for n in range(5):
    thissoln = np.zeros(shape=(1,101))
    den = np.random.uniform(0.01, 4, 2)
    t, yout = ct.step_response(ct.TransferFunction(den[1],den),t)
    thissoln[0] = yout
    thissoln=scaler.transform(thissoln)
    thissoln = torch.from_numpy(thissoln)
    thissoln = thissoln.to(torch.float32) 
    print(model.forward(thissoln))


