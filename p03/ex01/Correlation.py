import pandas as pd
import matplotlib.pyplot as plt

def main():
	train = pd.read_csv("../Train_knight.csv")
	train["knight_encoded"] = train["knight"].map({
		"Jedi": 1,
		"Sith": 0
	})
	correlations = train.corr(numeric_only=True, method="kendall")["knight_encoded"]

	correlations = correlations.sort_values(ascending=False)

	for col, correlation in correlations.items():
		name = "knight" if col == "knight_encoded" else col
		print(f"{name:<20} {correlation:>8.4f}")

if __name__ == "__main__":
	main()