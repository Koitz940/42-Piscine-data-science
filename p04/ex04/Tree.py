import pandas as pd
from collections import Counter
import random
import sys

class Tree_classifier:
	left: "Tree_classifier"
	right: "Tree_classifier"
	value: float
	column: str
	finish: bool
	eps: float
	prediction: str
	depth: int
	max_depth: int
	max_features: int

	def __init__(self, df, max_depth=10, eps=0.00000000000001, depth=0, max_features=5):
		self.max_depth = max_depth
		self.depth = depth
		self.eps = eps
		self.max_features = max_features
		self.init_params(df)

	def predict(self, sample):
		if self.finish:
			return self.prediction

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

		left_gini = Tree_classifier.gini_counts(left_jedi, left_sith)
		right_gini = Tree_classifier.gini_counts(right_jedi, right_sith)

		return (left_n * left_gini + right_n * right_gini) / total


	def init_params(self, df):
		labels = df["knight"].to_numpy()
		n = len(labels)

		self.prediction = Counter(labels).most_common(1)[0][0]

		if len(set(labels)) == 1 or n <= 1:
			self.finish = True
			return

		if self.depth >= self.max_depth:
			self.finish = True
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

				score = Tree_classifier.weighted_gini_counts(
					left_jedi, left_sith,
					right_jedi, right_sith
				)

				if score < best_gini:
					best_gini = score
					best_feature = feature
					best_threshold = (
						values[i] + values[i + 1]
					) / 2

		if best_feature is None:
			self.finish = True
			return

		self.column = best_feature
		self.value = best_threshold
		self.finish = False

		left = df[df[best_feature] <= best_threshold]
		right = df[df[best_feature] > best_threshold]

		self.left = Tree_classifier(
			left, self.max_depth, self.eps,
			depth=self.depth + 1,
			max_features=self.max_features
		)
		self.right = Tree_classifier(
			right, self.max_depth, self.eps,
			depth=self.depth + 1,
			max_features=self.max_features
		)

class Random_forest:

	def __init__(self, n_trees=100, max_features=5, max_depth=10, eps=0.00000000000001):
		self.trees = []
		self.n_trees = n_trees
		self.max_features = max_features
		self.eps = eps
		self.max_depth = max_depth

	def fit(self, df):
		for _ in range(self.n_trees):

			sample = df.sample(n=len(df), replace=True)

			tree = Tree_classifier(sample, max_depth=self.max_depth, eps=self.eps, depth=0, max_features=self.max_features)

			self.trees.append(tree)

	def predict(self, sample):
		predictions = [
			tree.predict(sample)
			for tree in self.trees
		]

		return max(
			set(predictions),
			key=predictions.count
        )

def main():
	if len(sys.argv) != 3:
		print("Bad imput, usage: python3 Tree.py train_file test_file")
		return
	
	train = 0
	test = 0

	with open(sys.argv[1]) as fd1:
		with open(sys.argv[2]) as fd2:
			train = pd.read_csv(fd1)
			test = pd.read_csv(fd2)

	values = {"Jedi": 0, "Sith": 1}

	forest = Random_forest(n_trees=100, max_features=5, max_depth=5)
	forest.fit(train)

	with open("Tree.txt", "w") as fd:
		predictions = []
		for _, row in test.iterrows():
			sample = row.to_dict()
			prediction = forest.predict(sample)
			predictions.append(prediction)

		fd.write("\n".join(predictions))


if __name__ == "__main__":
	main()