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

	fig, ax = plt.subplots(2, 1)
	
	ax[0].scatter(
		normalized_train[normalized_train["knight"] == "Jedi"]["Push"],
		normalized_train[normalized_train["knight"] == "Jedi"]["Deflection"],
		color="red",
		label="Jedi"
	)

	ax[0].scatter(
		normalized_train[normalized_train["knight"] == "Sith"]["Push"],
		normalized_train[normalized_train["knight"] == "Sith"]["Deflection"],
		color="blue",
		label="Sith"
	)

	ax[0].set_title("Train - Push vs Deflection")
	ax[0].set_xlabel("Push")
	ax[0].set_ylabel("Deflection")
	ax[0].legend()

	ax[1].scatter(
		normalized_test["Push"],
		normalized_test["Deflection"],
		color="green",
		label="Knight"
	)

	ax[1].set_title("Test - Push vs Deflection")
	ax[1].set_xlabel("Push")
	ax[1].set_ylabel("Deflection")
	ax[1].legend()

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()