import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

def freq_plot(df):
	print("Calculating amount of orders per user...")
	orders_per_user = df.groupby("user_id").size()

	print("Plotting amount of orders per user...")
	plt.hist(orders_per_user, bins=5, range=(0, 40), edgecolor="white")

	plt.xlim(0,40)
	plt.xlabel("Number of orders per user")
	plt.ylabel("Number of users")
	plt.title("Order Frequency")
	plt.show()

def money_plot(df):
	print("Calculating money spent per user...")
	money_per_user = df.groupby("user_id")["price"].sum()
	
	print("Plotting money spent per user...")
	plt.hist(money_per_user, bins=5, range=(-33, 250), edgecolor="white")

	plt.xlim(-40,250)
	plt.xlabel("Altarians per user")
	plt.ylabel("Number of users")
	plt.title("Altarians used")
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
		SELECT product_id, user_id, price
		FROM customers AS c
		WHERE c.event_type = 'purchase'
		""",
		conn
	)

	conn.close()

	freq_plot(df)
	money_plot(df)

if __name__ == "__main__":
	main()