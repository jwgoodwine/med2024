import csv
import matplotlib as mpl
mpl.use('pgf')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

width = 3.5*.9
height = width*(5**.5 - 1)/2
height = 1.1*height
plt.rcParams['figure.constrained_layout.use'] = True

fig, ax = plt.subplots(layout="constrained")

fig = plt.figure(figsize=(width,height))
for n in range(10):
  filename = "my_model_version_"+str(n)+".csv"
  with open(filename, newline='') as file:
    print(filename)
    reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)
    epoch=[]
    err = []
    for row in reader:
      print(row)
      epoch.append(row[1])
      err.append(row[2])

  plt.plot(epoch,err)

#plt.tight_layout()
plt.yscale("log")
plt.grid(True)
plt.xticks()
plt.yticks()
plt.xlabel("epoch")
plt.ylabel("Validation set error")
plt.savefig("overtrain.pgf",format="pgf")

