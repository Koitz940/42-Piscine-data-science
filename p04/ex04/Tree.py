import pandas as pd
from collections import Counter
import random
import sys

import matplotlib.pyplot as plt

def translate(l):
	return "Jedi" if l[0] > l[1] else "Sith"

class TreeClassifier:
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

class RandomForest:

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
		predictions = [
			tree.predict(sample)
			for tree in self.trees
		]

		ret = [sum(i[0] for i in predictions), sum(i[1] for i in predictions)]

		return ret

def plot_tree(root):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    def draw(node, x=0.5, y=1.0, dx=0.25, node_id="0"):
        if node is None:
            return

        # Main label
        if node.finish:
            label = f"Predict: {node.prediction}"
            color = "#d9ead3"
        else:
            label = f"{node.column} <= {node.value:.3f}"
            color = "#cfe2f3"

        # Add node statistics
        label += (
            f"\nGini: {node.gini:.3f}"
            f"\nSamples: {node.sample}"
            f"\nValues: {node.values}"
        )

        ax.text(
            x, y, label,
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=color,
                edgecolor="gray"
            ),
            fontsize=9
        )

        if node.finish:
            return

        children = [
            (node.left, x - dx, "Yes (≤)", "L"),
            (node.right, x + dx, "No (>)", "R")
        ]

        for child, child_x, edge_label, suffix in children:
            if child is None:
                continue

            child_y = y - 0.15

            ax.annotate(
                "",
                xy=(child_x, child_y + 0.025),
                xytext=(x, y - 0.025),
                arrowprops=dict(arrowstyle="->", color="gray")
            )

            ax.text(
                (x + child_x) / 2,
                (y + child_y) / 2,
                edge_label,
                fontsize=8,
                ha="center"
            )

            draw(
                child,
                child_x,
                child_y,
                dx / 2,
                node_id + suffix
            )

    draw(root)
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
    plt.show()

def describe(mat):
	tp = mat[0][0]
	fn = mat[0][1]
	fp = mat[1][0]
	tn = mat[1][1]

	total = tn + tp + fn + fp

	tprecision = tp / (tp + fp)
	trecall = tp / (tp + fn)

	nprecision = tn / (tn + fn)
	nrecall = tn / (tn + fp)

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
	
	forest = RandomForest(n_trees=100, max_features=5, max_depth=5)
	try:
		forest.fit(train)
	except:
		print("Something went wrong")

	try:
		for _, line in test.iterrows():
			truth.append(line["knight"])
			line = line.drop("knight")
			prediction.append(translate(forest.predict(line)))
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
		print("Bad imput, usage: python3 Tree.py train_file test_file")
		return
	
	train = 0
	test = 0

	try:
		with open(sys.argv[1]) as fd1:
			with open(sys.argv[2]) as fd2:
				train = pd.read_csv(fd1)
				test = pd.read_csv(fd2)
	except:
		print("failed to open file")
		return

	forest = RandomForest(n_trees=100, max_features=5, max_depth=5)
	try:
		forest.fit(train)
	except:
		print("Something went wrong")

	try:
		with open("Tree.txt", "w") as fd:
			predictions = []
			for _, row in test.iterrows():
				sample = row.to_dict()
				prediction = forest.predict(sample)
				predictions.append(translate(prediction))

			fd.write("\n".join(predictions))

		plot_tree(forest.trees[0])
	except:
		print("Something went wrong")


if __name__ == "__main__":
	main()