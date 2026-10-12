import pandas as pd
import matplotlib.pyplot as plt

def main():
	train = pd.read_csv("../Train_knight.csv")
	test = pd.read_csv("../Test_knight.csv")

	fig, axes = plt.subplots(6, 5, figsize=(18, 14))

	for i, column in enumerate(test):
		ax = axes[i // 5, i % 5]

		ax.hist(test[column], alpha=0.5,bins=40, label="Test")

		ax.set_title(column)
		ax.legend()

	plt.tight_layout()
	plt.show()

	fig, axes = plt.subplots(5, 6, figsize=(18, 14))

	for i, column in enumerate(train.columns[:-1]):
		ax = axes[i // 6, i % 6]

		ax.hist(
			train[train["knight"] == "Sith"][column],
			bins=40,
			alpha=0.5,
			label="Sith"
		)

		ax.hist(
			train[train["knight"] == "Jedi"][column],
			bins=40,
			alpha=0.5,
			label="Jedi"
		)

		ax.set_title(column)
		ax.legend()

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	main()