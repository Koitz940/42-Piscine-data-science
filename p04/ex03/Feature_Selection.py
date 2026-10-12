from sklearn.linear_model import LinearRegression
import pandas as pd

def get_vifs(df):
	vifs = {}

	for column in df.columns:
		y = df[column]
		X = df.drop(columns=[column])

		model = LinearRegression()
		model.fit(X, y)

		r2 = model.score(X, y)
		vifs[column] = 1 / (1 - r2)

	return vifs

def get_max_key(d):
    return max(d, key=d.get)

def remove_until_thing(df):
	vifs = get_vifs(df)
	while max(vifs.values()) > 5:
		m = get_max_key(vifs)
		df = df.drop(columns=[m])
		print(f"removing: {m} with a VIF of {vifs[m]}")
		vifs = get_vifs(df)
	print(df)
	print("\nVIFs of them:")
	for key in vifs:
		print(f"{key}: {vifs[key]:.2f}, so a tolerance of {1/vifs[key]:.2f}")
		

def main():
	train = pd.read_csv("../Train_knight.csv")
	train = train.drop(columns=["knight"])
	remove_until_thing(train)

if __name__ == "__main__":
	main()