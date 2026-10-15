import abc
import pandas as pd
from collections import Counter
import math
import sys
import numpy as np
import random
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler

def translate(l):
	return "Jedi" if l[0] > l[1] else "Sith"

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


class Classifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, sample) -> list[float]:
        pass


class TreeClassifier(Classifier):
	left: "TreeClassifier"
	right: "TreeClassifier"
	value: float
	column: str
	finish: bool
	depth: int
	max_depth: int
	max_features: int
	gini: float
	sample: int
	values: list[int]
	prediction: str
	features: list[str]

	def __init__(self, df, max_depth=10, depth=0, max_features=5):
		self.max_depth = max_depth
		self.depth = depth
		self.max_features = max_features
		self.gini = 0
		self.features = list(df.columns.drop("knight"))
		self.sample = 1
		self.init_params(df)

	def predict(self, sample):
		if isinstance(sample, np.ndarray):
			sample = dict(zip(self.features, sample.flatten()))

		if self.finish:
			return [i / self.sample  for i in self.values]

		cur_answer = sample[self.column] <= self.value

		if cur_answer:
			return self.left.predict(sample)
		return self.right.predict(sample)


	@staticmethod
	def gini_counts(jedi, sith):
		total = jedi + sith
		if total == 0:
			return 0

		return 1 - (jedi / total) * (jedi / total) - (sith / total) * (sith / total)


	@staticmethod
	def weighted_gini_counts(left_jedi, left_sith, right_jedi, right_sith):
		left_n = left_jedi + left_sith
		right_n = right_jedi + right_sith
		total = left_n + right_n

		if left_n == 0 or right_n == 0:
			return float("inf")

		left_gini = TreeClassifier.gini_counts(left_jedi, left_sith)
		right_gini = TreeClassifier.gini_counts(right_jedi, right_sith)

		return (left_n * left_gini + right_n * right_gini) / total


	def init_params(self, df):
		labels = df["knight"].to_numpy()
		n = len(labels)
		self.sample = n

		c = Counter(labels)
		self.values = [c["Jedi"], c["Sith"]]
		self.prediction = translate(self.values)

		if self.values[0] == 0:
			self.finish = True
			return

		if self.values[1] == 0:
			self.finish = True
			return

		if self.depth >= self.max_depth:
			self.finish = True
			self.gini = self.gini_counts(self.values[0], self.values[1])
			return

		features = list(df.columns.drop("knight"))

		if self.max_features is not None:
			features = random.sample(
				features,
				min(self.max_features, len(features))
			)

		best_gini = float("inf")
		best_feature = None
		best_threshold = None

		for feature in features:
			order = df[feature].to_numpy().argsort()
			values = df[feature].to_numpy()[order]
			sorted_labels = labels[order]

			right_jedi = sum(sorted_labels == "Jedi")
			right_sith = sum(sorted_labels == "Sith")
			left_jedi = 0
			left_sith = 0

			for i in range(n - 1):
				if sorted_labels[i] == "Jedi":
					left_jedi += 1
					right_jedi -= 1
				else:
					left_sith += 1
					right_sith -= 1

				if values[i] == values[i + 1]:
					continue

				score = TreeClassifier.weighted_gini_counts(
					left_jedi, left_sith,
					right_jedi, right_sith
				)

				if score < best_gini:
					best_gini = score
					best_feature = feature
					best_threshold = (
						values[i] + values[i + 1]
					) / 2

		self.gini = best_gini

		if best_feature is None:
			self.finish = True
			return

		self.column = best_feature
		self.value = best_threshold
		self.finish = False

		left = df[df[best_feature] <= best_threshold]
		right = df[df[best_feature] > best_threshold]

		self.left = TreeClassifier(
			left, self.max_depth,
			depth=self.depth + 1,
			max_features=self.max_features
		)
		self.right = TreeClassifier(
			right, self.max_depth,
			depth=self.depth + 1,
			max_features=self.max_features
		)


class RandomForest(Classifier):

    def __init__(self, n_trees=100, max_features=5, max_depth=10):
        self.trees = []
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth

    def fit(self, df):
        for _ in range(self.n_trees):

            sample = df.sample(n=len(df), replace=True)

            tree = TreeClassifier(sample, max_depth=self.max_depth, depth=0, max_features=self.max_features)

            self.trees.append(tree)

    def predict(self, sample):
        predictions = [tree.predict(sample) for tree in self.trees]

        ret = [sum(i[0] for i in predictions), sum(i[1] for i in predictions)]

        s = ret[0] + ret[1]

        return [i / s for i in ret]


class KNN(Classifier):
	def __init__(self, k, df, distance):
		self.k = k
		self.distance = distance

		self.features = df.columns.drop("knight")
		self.X = df[self.features].to_numpy(dtype=float)

		self.y = df["knight"].to_numpy()

		self.candidates = [
			dict(zip(self.features, row))
			for row in self.X
		]

	def predict(self, sample):
		if isinstance(sample, np.ndarray):
			sample = dict(zip(self.features, sample.flatten()))

		sample = {
			key: value
			for key, value in sample.items()
		}

		distances = np.fromiter(
			(
				self.distance(sample, candidate)
				for candidate in self.candidates
			),
			dtype=float,
			count=len(self.candidates)
		)

		k = min(self.k, len(distances))
		nearest_idx = np.argpartition(distances, k - 1)[:k]

		votes = Counter()

		for i in nearest_idx:
			votes[self.y[i]] += 1 / (distances[i] + 1e-10)

		s = votes["Jedi"] + votes["Sith"]

		return [votes["Jedi"] / s, votes["Sith"] / s]


class VotingClassifier(Classifier):
	classifiers: list[Classifier]
	weights = list[float]

	def __init__(self, classifiers, weights=None):
		self.classifiers = classifiers
		if weights is None:
			self.weights = [1 for _ in self.classifiers]

		else:
			if len(weights) != len(self.classifiers):
				raise ValueError("Weights list must have the same length as the classifiers list")
			self.weights = [float(i) for i in weights]

	def predict(self, sample):
		ret = [0, 0]
		for (classifier, weight) in zip(self.classifiers, self.weights):
			pred = classifier.predict(sample)
			ret[0] += pred[0] * weight
			ret[1] += pred[1] * weight

		s = ret[0] + ret[1]

		return [i / s for i in ret]


class MLP(Classifier):
	learning_rate: float
	layers: list[int]
	weights: list[list[float]]
	biases: list[list[float]]

	def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.1, seed=None):
		self.learning_rate = learning_rate
		rng = np.random.default_rng(seed)

		self.layers = [input_size] + hidden_layers + [output_size]

		self.weights = []
		self.biases = []

		for fan_in, fan_out in zip(self.layers[:-1], self.layers[1:]):
			W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(
				2.0 / (fan_in + fan_out)
			)
			b = np.zeros((1, fan_out))

			self.weights.append(W)
			self.biases.append(b)

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

	def softmax(self, z):
		exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
		return exp_z / np.sum(exp_z, axis=1, keepdims=True)

	def forward(self, X):
		self.activations = [X]
		self.z_values = []

		for i in range(len(self.weights) - 1):
			z = self.activations[-1] @ self.weights[i] + self.biases[i]
			a = self.sigmoid(z)

			self.z_values.append(z)
			self.activations.append(a)

		z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
		a = self.softmax(z)

		self.z_values.append(z)
		self.activations.append(a)

		return a

	def cross_entropy(self, y_true, y_pred):
		eps = 1e-12
		y_pred = np.clip(y_pred, eps, 1.0)
		return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

	def backward(self, y_true):
		m = y_true.shape[0]

		delta = self.activations[-1] - y_true

		weight_grads = [None] * len(self.weights)
		bias_grads = [None] * len(self.biases)

		for i in reversed(range(len(self.weights))):
			weight_grads[i] = self.activations[i].T @ delta / m
			bias_grads[i] = np.sum(delta, axis=0, keepdims=True) / m

			if i > 0:
				delta = (
					delta @ self.weights[i].T
					* self.activations[i]
					* (1 - self.activations[i])
				)

		for i in range(len(self.weights)):
			self.weights[i] -= self.learning_rate * weight_grads[i]
			self.biases[i] -= self.learning_rate * bias_grads[i]

	def fit(self, X, y, epochs=1000, verbose=False):
		for epoch in range(epochs):
			predictions = self.forward(X)
			loss = self.cross_entropy(y, predictions)

			self.backward(y)

			if verbose and (epoch + 1) % 100 == 0:
				print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")

	def predict(self, X):
		result = self.forward(np.asarray(X, dtype=float))

		if result.shape[0] == 1:
			return result[0].tolist()

		return result


def scale(train, test):
    features = train.columns.drop("knight")

    X_train = train[features].copy()
    X_test = test[features].copy()

    y_labels = train["knight"].map({
        "Jedi": 0,
        "Sith": 1
    }).to_numpy()

    y_train = np.eye(2)[y_labels]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train.astype(float), X_test.astype(float), y_train

def prepare_mlp(X_train, y_train):
	model = MLP(
		input_size=X_train.shape[1],
		hidden_layers=[10, 16],
		output_size=2,
		learning_rate=0.1
	)

	model.fit(X_train, y_train, epochs=1000, verbose=True)
	return model

def do(train, test):
	print("Scaling down the data")
	x_train, x_test, y_train = scale(train, test)
	train_scaled = pd.DataFrame(
		x_train,
		columns=train.drop(columns=["knight"]).columns,
		index=train.index
	)
	train_scaled["knight"] = train["knight"]

	print("Training Random Forest")
	forest = RandomForest()
	forest.fit(train_scaled)

	print("Training KNN")
	knn = KNN(5, train_scaled, cosine_distance)

	print("Training MLP")
	mlp = prepare_mlp(x_train, y_train)

	model = VotingClassifier([mlp, knn, forest], [10, 3, 1])
	
	with open("Voting.txt", "w") as fd:
		predictions = []
		for row in x_test:
			prediction = model.predict(row)
			predictions.append(translate(prediction))

		fd.write("\n".join(predictions))

def describe(mat):
	tp = mat[0][0]
	fn = mat[0][1]
	fp = mat[1][0]
	tn = mat[1][1]

	total = tn + tp + fn + fp

	tprecision = 0
	try:
		tprecision = tp / (tp + fp)
	except:
		tprecision = 1

	trecall = 0
	try:
		trecall = tp / (tp + fn)
	except:
		trecall = 1

	nprecision = 0
	try:
		nprecision = tn / (tn + fn)
	except:
		nprecision = 1

	nrecall = 0

	try:
		nrecall = tn / (tn + fp)
	except:
		nrecall = 1

	tf1 = 2 * tprecision * trecall / (tprecision + trecall)
	nf1 = 2 * nprecision * nrecall / (nprecision + nrecall)

	acc = (tp + tn) / total

	print(f"{'':10}{'precision':10}{'recall':10}{'f1-score':10}{'total':10}\n")
	print(f"{'Jedi':10}{tprecision:<10.2f}{trecall:<10.2f}{tf1:<10.2f}{sum(mat[0]):<10}")
	print(f"{'Sith':10}{nprecision:<10.2f}{nrecall:<10.2f}{nf1:<10.2f}{sum(mat[1]):<10}\n")
	print(f"{'accuracy':10}{'':10}{'':10}{acc:<10.2f}{total:<10}\n")
	print(f"[[{tp} {fp}]\n [{fn} {tn}]]")

def evaluate():
	truth = []
	prediction = []
	train = 0
	test = 0

	try:
		with open("../Training_knight.csv") as fd1:
			with open("../Validation_knight.csv") as fd2:
				train = pd.read_csv(fd1)
				test = pd.read_csv(fd2)
	except:
		print("Something went wrong")
		return

	new_test = test.drop(columns=["knight"])

	x_train, x_test, y_train = scale(train, new_test)
	
	train_scaled = pd.DataFrame(
		x_train,
		columns=train.drop(columns=["knight"]).columns,
		index=train.index
	)
	train_scaled["knight"] = train["knight"]

	test_scaled = pd.DataFrame(
			x_test,
			columns=test.drop(columns=["knight"]).columns,
			index=test.index
		)
	test_scaled["knight"] = test["knight"]

	print("Training Random Forest")
	forest = RandomForest()
	forest.fit(train_scaled)

	print("Training KNN")
	knn = KNN(5, train_scaled, cosine_distance)

	print("Training MLP")
	mlp = prepare_mlp(x_train, y_train)

	model = VotingClassifier([mlp, knn, forest], [2, 2, 1.5])

	try:
		for _, line in test_scaled.iterrows():
			truth.append(line["knight"])
			line = line.drop("knight")
			prediction.append(translate(model.predict(line)))
	except:
		print("Something went wrong")

	vals = {"Jedi": 0, "Sith": 1}
	mat = [[0, 0], [0, 0]]
	for t, p in zip(truth, prediction):
		val1 = vals[t]
		val2 = vals[p]
		mat[val1][val2] += 1
	describe(mat)

def main():
	if len(sys.argv) != 3:
			print("Bad imput, usage: python3 KNN.py train_file test_file")
			return
		
	with open(sys.argv[1]) as fd1:
		with open(sys.argv[2]) as fd2:
			train = pd.read_csv(fd1)
			test = pd.read_csv(fd2)
			do(train, test)


if __name__ == "__main__":
	main()