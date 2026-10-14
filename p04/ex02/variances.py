import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
	train = pd.read_csv("../Train_knight.csv")
	train = train.drop(columns=["knight"])

	scaler = StandardScaler()
	X = scaler.fit_transform(train)

	pca = PCA()
	pca.fit(X)

	print("Variances (percentage):")
	print(pca.explained_variance_ratio_)

	print("\nCumulative varaince (percentage)")
	cumsum = pca.explained_variance_ratio_.cumsum() * 100

	print(cumsum)
	values = [0] + cumsum.tolist()

	plt.plot(range(len(values)), values, marker=".")
	plt.xlabel("Number of components")
	plt.ylabel("Explained Variance (%)")
	plt.xlim(0, 32)
	plt.ylim(40, 105)
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
    main()