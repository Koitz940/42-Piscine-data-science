import pandas as pd
import matplotlib.pyplot as plt

def main():
	train = pd.read_csv("../Train_knight.csv")
	test = pd.read_csv("../Test_knight.csv")
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))

	axes[0, 0].scatter(
		train[train["knight"] == "Jedi"]["Empowered"],
		train[train["knight"] == "Jedi"]["Sprint"],
		color="red",
		label="Jedi"
	)

	axes[0, 0].scatter(
		train[train["knight"] == "Sith"]["Empowered"],
		train[train["knight"] == "Sith"]["Sprint"],
		color="blue",
		label="Sith"
	)

	axes[0, 0].set_title("Train - Empowered vs Sprint")
	axes[0, 0].set_xlabel("Empowered")
	axes[0, 0].set_ylabel("Sprint")
	axes[0, 0].legend()

	axes[0, 1].scatter(
		train[train["knight"] == "Jedi"]["Push"],
		train[train["knight"] == "Jedi"]["Deflection"],
		color="red",
		label="Jedi"
	)

	axes[0, 1].scatter(
		train[train["knight"] == "Sith"]["Push"],
		train[train["knight"] == "Sith"]["Deflection"],
		color="blue",
		label="Sith"
	)

	axes[0, 1].set_title("Train - Push vs Deflection")
	axes[0, 1].set_xlabel("Push")
	axes[0, 1].set_ylabel("Deflection")
	axes[0, 1].legend()

	axes[1, 0].scatter(
		test["Empowered"],
		test["Sprint"],
		color="green"
	)

	axes[1, 0].set_title("Test - Empowered vs Sprint")
	axes[1, 0].set_xlabel("Empowered")
	axes[1, 0].set_ylabel("Sprint")

	axes[1, 1].scatter(
		test["Push"],
		test["Deflection"],
		color="green"
	)

	axes[1, 1].set_title("Test - Push vs Deflection")
	axes[1, 1].set_xlabel("Push")
	axes[1, 1].set_ylabel("Deflection")


	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()