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

	fig, ax = plt.subplots()

	ax.scatter(
		standardized_train[standardized_train["knight"] == "Jedi"]["Empowered"],
		standardized_train[standardized_train["knight"] == "Jedi"]["Sprint"],
		color="red",
		label="Jedi"
	)

	ax.scatter(
		standardized_train[standardized_train["knight"] == "Sith"]["Empowered"],
		standardized_train[standardized_train["knight"] == "Sith"]["Sprint"],
		color="blue",
		label="Sith"
	)

	ax.set_title("Train - Empowered vs Sprint")
	ax.set_xlabel("Empowered")
	ax.set_ylabel("Sprint")
	ax.legend()

	plt.show()

if __name__ == "__main__":
	main()