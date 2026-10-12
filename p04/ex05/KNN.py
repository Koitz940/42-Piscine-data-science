import pandas as pd
from collections import Counter
from collections.abc import Callable
import math
import sys
import matplotlib.pyplot as plt

def euclidean(s1, s2):
	return math.sqrt(sum((a - s2[key]) * (a - s2[key]) for (key, a) in s1.items()))

def manhattan(s1, s2):
    return sum(abs(s1[k] - s2[k]) for k in s1)


def chebyshev(s1, s2):
    return max(abs(s1[k] - s2[k]) for k in s1)


def minkowski(s1, s2, p=3):
    return sum(
        abs(s1[k] - s2[k]) ** p for k in s1
    ) ** (1 / p)


def cosine_distance(s1, s2):
    dot = sum(s1[k] * s2[k] for k in s1)

    norm1 = math.sqrt(sum(s1[k] * s1[k] for k in s1))
    norm2 = math.sqrt(sum(s2[k] * s2[k] for k in s2))

    denominator = norm1 * norm2

    if denominator == 0:
        return 1.0 if norm1 != norm2 else 0.0

    return 1 - dot / denominator


def canberra(s1, s2):
    total = 0.0

    for k in s1:
        denominator = abs(s1[k]) + abs(s2[k])

        if denominator != 0:
            total += abs(s1[k] - s2[k]) / denominator

    return total

class KNN:
	mean: float
	k: int
	df: pd.DataFrame
	sd: float
	
	def __init__(self, k, df, distance):
		self.k = k
		self.distance = distance

		self.features = df.columns.drop("knight")
		self.mean = df[self.features].mean()
		self.std = df[self.features].std().replace(0, 1)

		self.df = df.copy()
		self.df[self.features] = (df[self.features] - self.mean) / self.std

	def predict(self, sample):
		sample = {
			key: (value - self.mean[key]) / self.std[key]
			for key, value in sample.items()
		}

		nearest = []

		for _, row in self.df.iterrows():
			label = row["knight"]
			candidate = row[self.features].to_dict()
			d = self.distance(sample, candidate)

			i = 0
			while i < len(nearest) and nearest[i][0] < d:
				i += 1

			nearest.insert(i, (d, label))

			if len(nearest) > self.k:
				nearest.pop()

		votes = Counter()
		for d, label in nearest:
			votes[label] += 1 / (d + 1e-10)

		return votes.most_common(1)[0][0]
	
def describe(mat):
	tprecision = mat[0][0] / (mat[0][0] + mat[0][1])
	trecall = mat[0][0] / (mat[0][0] + mat[1][0])
	nprecision = mat[1][1] / (mat[1][1] + mat[1][0])
	nrecall = mat[1][1] / (mat[1][1] + mat[0][1])
	tf1 = 2 * tprecision * trecall / (tprecision + trecall)
	nf1 = 2 * nprecision * nrecall / (nprecision + nrecall)
	acc = (tf1 + nf1) / 2
	total = sum(sum(l) for l in mat)
	print(f"{'':10}{'precision':10}{'recall':10}{'f1-score':10}{'total':10}\n")
	print(f"{'Jedi':10}{tprecision:<10.2f}{trecall:<10.2f}{tf1:<10.2f}{sum(mat[0]):<10}")
	print(f"{'Sith':10}{nprecision:<10.2f}{nrecall:<10.2f}{nf1:<10.2f}{sum(mat[1]):<10}\n")
	print(f"{'accuracy':10}{'':10}{'':10}{acc:<10.2f}{total:<10}\n")
	print(f"[[{mat[0][0]} {mat[0][1]}]\n [{mat[1][0]} {mat[1][1]}]]")
	return acc


def test_distances():
	values = {"Jedi": 0, "Sith": 1}
	ks = []

	train = pd.read_csv("../Training_knight.csv")
	test = pd.read_csv("../Validation_knight.csv")

	for k in range(1, 31):
		mat = [[0, 0], [0, 0]]
		model = KNN(k, train, cosine_distance)
		for _, row in test.iterrows():
			sample = row.drop("knight").to_dict()
			prediction = model.predict(sample)
			mat[values[prediction]][values[row["knight"]]] += 1
		ks.append(describe(mat))


	plt.plot(range(1,31), ks)
	plt.show()

def do(train, test):
	knn = KNN(5, train, cosine_distance)

	with open("KNN.txt", "w") as fd:
		predictions = []
		for _, row in test.iterrows():
			sample = row.to_dict()
			prediction = knn.predict(sample)
			predictions.append(prediction)

		fd.write("\n".join(predictions))

def main():
	if len(sys.argv) != 3:
		print("Bad imput, usage: python3 KNN.py train_file test_file")
		return
	
	with open(sys.argv[1]) as fd1:
		with open(sys.argv[2]) as fd2:
			train = pd.read_csv(fd1)
			test = pd.read_csv(fd2)
			do(train, test)
			test_distances()



if __name__ == "__main__":
	main()