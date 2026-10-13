import pandas as pd
import matplotlib.pyplot as plt

def main():
	train = pd.read_csv("../Train_knight.csv")
	test = pd.read_csv("../Test_knight.csv")

	standardized_train = train.copy()

	features = train.columns.drop("knight")

	mean = train[features].mean()
	std = train[features].std()

	standardized_train[features] = (train[features] - mean) / std

	mean = test[features].mean()
	std = test[features].std()
	standardized_test = (test[features] - mean) / std

	print("standardized train:")
	print(standardized_train)
	print("standardized test:")
	print(standardized_test)

	fig, ax = plt.subplots(2,1)

	ax[0].scatter(
		standardized_train[standardized_train["knight"] == "Jedi"]["Empowered"],
		standardized_train[standardized_train["knight"] == "Jedi"]["Sprint"],
		color="red",
		label="Jedi"
	)

	ax[0].scatter(
		standardized_train[standardized_train["knight"] == "Sith"]["Empowered"],
		standardized_train[standardized_train["knight"] == "Sith"]["Sprint"],
		color="blue",
		label="Sith"
	)

	ax[0].set_title("Train - Empowered vs Sprint")
	ax[0].set_xlabel("Empowered")
	ax[0].set_ylabel("Sprint")
	ax[0].legend()

	ax[1].scatter(
		standardized_test["Empowered"],
		standardized_test["Sprint"],
		color="green",
		label="Knight"
	)

	ax[1].set_title("Test - Empowered vs Sprint")
	ax[1].set_xlabel("Empowered")
	ax[1].set_ylabel("Sprint")
	ax[1].legend()

	plt.show()

if __name__ == "__main__":
	main()