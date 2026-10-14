import abc

import pandas as pd
from collections import Counter
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from abc import ABC, abstractmethod

def translate(l):
	return "Jedi" if l[0] > l[1] else "Sith"

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

	def __init__(self, df, max_depth=10, depth=0, max_features=5):
		self.max_depth = max_depth
		self.depth = depth
		self.max_features = max_features
		self.gini = 0
		self.sample = 1
		self.init_params(df)

	def predict(self, sample):
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

class KNN(Classifier):
	def __init__(self, k, df, distance):
		self.k = k
		self.distance = distance

		self.features = df.columns.drop("knight")
		self.mean = df[self.features].mean()
		self.std = df[self.features].std().replace(0, 1)

		self.X = (
			(df[self.features] - self.mean) / self.std
		).to_numpy(dtype=float)

		self.y = df["knight"].to_numpy()

		self.candidates = [
			dict(zip(self.features, row))
			for row in self.X
		]

	def predict(self, sample):
		sample = {
			key: (value - self.mean[key]) / self.std[key]
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

		s = float(votes["Jedi"]) + float(votes["Sith"])
		return [float(votes["Jedi"]) / s, float(votes["Sith"]) / s]


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


