import pandas as pd
import matplotlib.pyplot as plt

def main():
	train = pd.read_csv("../Train_knight.csv")
	test = pd.read_csv("../Test_knight.csv")

	normalized_train = train.copy()

	features = train.columns.drop("knight")

	mins = train[features].min()
	maxs = train[features].max()

	normalized_train[features] = (train[features] - mins) / (maxs - mins)

	mins = test[features].min()
	maxs = test[features].max()
	normalized_test = (test[features] - mins) / (maxs - mins)

	print("normalized train:")
	print(normalized_train)
	print("normalized test:")
	print(normalized_test)

	fig, ax = plt.subplots()
	
	ax.scatter(
		normalized_train[normalized_train["knight"] == "Jedi"]["Push"],
		normalized_train[normalized_train["knight"] == "Jedi"]["Deflection"],
		color="red",
		label="Jedi"
	)

	ax.scatter(
		normalized_train[normalized_train["knight"] == "Sith"]["Push"],
		normalized_train[normalized_train["knight"] == "Sith"]["Deflection"],
		color="blue",
		label="Sith"
	)


	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()