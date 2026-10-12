import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def plot_clusters(clusters, scaler_before, scaler_february, kmeans_before, kmeans_february, cluster_names):
	plt.figure(figsize=(12, 7))

	colors = ["#504F4E", "#D11313", "#D6BF39", "#0B2EF1"]

	print("Showing each subgroup in 2D plot of money spent Vs Latest purchase...")
	for i, cluster_df in enumerate(clusters):
		plt.scatter(
			cluster_df["total_price"],
			cluster_df["event_days"],
			color=colors[i],
			label=cluster_names[i],
			alpha=0.6,
			s=35
		)

	centroids_before = scaler_before.inverse_transform(
		kmeans_before.cluster_centers_
	)
	centroids_february = scaler_february.inverse_transform(
		kmeans_february.cluster_centers_
	)

	for i, centroid in enumerate(centroids_before):
		plt.scatter(
			centroid[0], centroid[1],
			marker="X",
			s=250,
			color=colors[i],
			edgecolor="black",
			linewidth=1.5,
			zorder=5
		)

	for i, centroid in enumerate(centroids_february):
		plt.scatter(
			centroid[0], centroid[1],
			marker="X",
			s=250,
			color=colors[i + 2],
			edgecolor="black",
			linewidth=1.5,
			zorder=5
		)

	plt.xlabel("Total Price")
	ax = plt.gca()
	ax.set_ylabel("Latest Purchase Date")
	ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda y, _: pd.to_datetime(
            y * 1000, unit="D", origin="unix"
        ).strftime("%b %Y")
    )
)
	plt.tight_layout()
	plt.legend()
	plt.show()


def plot_cluster_sizes(clusters, cluster_names):
	print("Calculating amount of customers in each subgroup")
	counts = [cluster["total_price"].sum() for cluster in clusters]

	print("plotting amount of customers on each group")
	plt.figure(figsize=(11, 6))

	bars = plt.bar(
		cluster_names,
		counts,
		color=["#504F4E", "#D11313", "#D6BF39", "#0B2EF1"]
	)

	plt.bar_label(bars, fmt="%.0f")
	plt.xlabel("Customer segment")
	plt.ylabel("Number of customers")
	plt.title("Customer Distribution by Segment")
	plt.xticks(rotation=15, ha="right")
	plt.tight_layout()
	plt.show()


def run_kmeans(data, n_clusters):
    data = data.copy()

    data["event_days"] = (
        data["latest_event_time"].astype("int64") / 86_400_000_000_000
    )

    features = data[["total_price", "event_days"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    data["cluster"] = kmeans.fit_predict(X_scaled)

    return data, kmeans, scaler
	

def cluster_customers(df):
	print("Separating customers by latest purchase...")
	df["latest_event_time"] = pd.to_datetime(df["latest_event_time"])

	february_2023 = df[
		(df["latest_event_time"] >= "2023-02-01") &
		(df["latest_event_time"] < "2023-03-01")
	]

	before_february_2023 = df[
		df["latest_event_time"] < "2023-02-01"
	]

	print("Separating inactive customers with K-means... (2)")
	before_february_clustered, kmeans_before, scaler_before = run_kmeans(
		before_february_2023, n_clusters=2
	)

	print("Separating active customers with K-means... (2)")
	february_clustered, kmeans_february, scaler_february = run_kmeans(
		february_2023, n_clusters=2
	)

	clusters = [
		before_february_clustered[before_february_clustered["cluster"] == 0],
		before_february_clustered[before_february_clustered["cluster"] == 1],
		february_clustered[february_clustered["cluster"] == 0],
		february_clustered[february_clustered["cluster"] == 1],
	]

	cluster_names = ["Inactive", "Previous Spender", "Gold user", "Regular"]
	print(before_february_2023["latest_event_time"].min())
	plot_cluster_sizes(clusters, cluster_names)
	plot_clusters(clusters, scaler_before, scaler_february, kmeans_before, kmeans_february, cluster_names)


def main():
	print("Connecting...")
	conn = psycopg2.connect(
		host="localhost",
		database="piscineds",
		user="gcassi-d",
		password="mysecretpassword"
	)

	print("Reading postgres...")
	query = """
	SELECT
		user_id,
		SUM(price) AS total_price,
		MAX(event_time) AS latest_event_time
	FROM customers AS c
    WHERE c.event_type = 'purchase'
	GROUP BY user_id
	"""

	df = pd.read_sql(query, conn)

	conn.close()

	cluster_customers(df)


if __name__ == "__main__":
    main()