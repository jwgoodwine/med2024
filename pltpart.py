
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

