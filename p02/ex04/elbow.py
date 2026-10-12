import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def elbow(df):

	scaler = StandardScaler()

	X = scaler.fit_transform(df)

	inertias = []

	print("Calculating inertias...")
	for k in range(1, 11):
		kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
		kmeans.fit(X)
		print(f"k = {k}: inertia: {kmeans.inertia_}")
		inertias.append(kmeans.inertia_)

	print("plotting...")
	plt.plot(range(1, 11), inertias, marker="o")
	plt.xlabel("Number of clusters")
	plt.ylabel("Inertia")
	plt.title("Elbow Method")
	plt.show()


def main():
	print("connecting...")
	conn = psycopg2.connect(
		host="localhost",
		database="piscineds",
		user="gcassi-d",
		password="mysecretpassword"
	)

	print("Reading postgres...")
	df = pd.read_sql(
	"""
		SELECT user_id,
		sum(price)		AS monetary,
			count(*)	 AS frequency,
			EXTRACT(DAY FROM (DATE '2023-03-01' - max(event_time))) AS recency
		FROM customers
		WHERE event_type = 'purchase'
		GROUP BY user_id;
	""",
		conn
	)

	conn.close()
	elbow(df)

if __name__ == "__main__":
	main()
	