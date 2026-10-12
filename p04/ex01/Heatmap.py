import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import math

def mean(l):
	return sum(l) / len(l)

def sd(l, mean):
	return math.sqrt(sum((i - mean) * (i - mean) for i in l) / len(l))

def correlation(l1, l2, sd1, sd2, mean1, mean2):
	return sum((i - mean1) * (j - mean2) for i, j in zip(l1, l2)) / (len(l1) * sd1 * sd2)

def get_coefs(df):
	properties = df.columns.tolist()
	df_list = [df[i].tolist() for i in properties]
	means = [mean(i) for i in df_list]
	sds = [sd(i, m) for i, m in zip(df_list, means)]
	coefs = [[correlation(l1, l2, sd1, sd2, m1, m2) for l1, sd1, m1 in zip(df_list, sds, means)] for l2, sd2, m2 in zip(df_list, sds, means)]
	return coefs, properties

def plot(coefs, properties):
	cmap = LinearSegmentedColormap.from_list(
		"black_red_white",
		["black", "red", "white"]
	)

	plt.imshow(coefs, cmap=cmap)

	plt.colorbar()

	plt.xticks(range(len(properties)), properties, rotation=90)
	plt.yticks(range(len(properties)), properties)

	plt.xlabel("Predicted")
	plt.ylabel("Actual")

	plt.show()

def main():
	train = pd.read_csv("../Train_knight.csv")
	train["knight"] = train["knight"].map({"Jedi": 0, "Sith": 1})
	coefs, properties = get_coefs(train)
	plot(coefs, properties)

if __name__ == "__main__":
	main()
