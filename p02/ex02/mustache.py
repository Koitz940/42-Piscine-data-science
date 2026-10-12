import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

def boxplots(df):
	price = df["price"]
	print(df["price"].describe())

	print("Plotting purchase prices...")
	plt.boxplot(
		price,
		vert=False,
		patch_artist=True,
		widths=2,
		showfliers=False,
		medianprops=dict(color="black"),
		boxprops=dict(facecolor="green")
	)
	plt.xlabel("price")
	plt.show()

	plt.boxplot(price, widths=2, vert=False, boxprops=dict(color="none"), medianprops=dict(color="black"))
	plt.xlabel("price")
	plt.show()

def other_plot(df):
	print("Calculating average basquet price per customer...")
	average_basket_price = df.groupby("user_id")["price"].mean()

	print("Plotting average basquet price per customer...")
	plt.boxplot(average_basket_price, 
	        vert=False,
			patch_artist=True,
			widths=2,
			medianprops=dict(color="black"),
			showfliers=False,
			boxprops=dict(facecolor="lightblue"))

	plt.xlabel("Average basket price")
	plt.title("Average Basket Price per User")

	plt.show()


def main():
	print("Connecting...")
	conn = psycopg2.connect(
		host="localhost",
		database="piscineds",
		user="gcassi-d",
		password="mysecretpassword"
	)

	print("Reading postgres...")
	df = pd.read_sql(
		"""
		SELECT event_time, user_id, event_type, price
		FROM customers AS c
		WHERE c.event_type = 'purchase'
		""",
		conn
	)

	conn.close()

	boxplots(df)
	other_plot(df)


if __name__ == "__main__":
	main()