import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

def plot_daily_customers(df):
	print("Calculating amount of daily customers...")
	daily_customers = df.groupby(df["event_time"].dt.date)["user_id"].nunique()
	print("Printing amount of daily customers...")
	daily_customers.plot()
	plt.ylabel('Number of customers')
	plt.show()

def plot_sales_barplot(df):
	print("Calculating amount of monthly revenue...")
	monthly_revenue = df.groupby(df["event_time"].dt.to_period("M"))["price"].sum()
	monthly_revenue = monthly_revenue / 1000000

	print("Plotting amount of monthly revenue...")
	plt.figure(figsize=(10, 6))
	plt.bar(
		monthly_revenue.index.astype(str),
		monthly_revenue.values
	)

	plt.xlabel("Month")
	plt.ylabel("Revenue (millions)")
	plt.title("Monthly Revenue")

	plt.ylim(0, monthly_revenue.max() * 1.1)

	plt.show()


def plot_spending_per_customer(df):
	print("Calculating average spending per customer...")
	daily_customers = df.groupby(df["event_time"].dt.date)["user_id"].nunique()
	daily_revenue = df.groupby(df["event_time"].dt.date)["price"].sum()
	average = daily_revenue / daily_customers

	print("Plotting average spending per customer...")
	plt.plot(average)
	plt.fill_between(
		average.index,
		average.values,
		alpha=0.3
	)

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

	print("transforming data...")
	df["event_time"] = pd.to_datetime(
		df["event_time"],
		format="%Y-%m-%d %H:%M:%S"
	)

	plot_daily_customers(df)
	plot_sales_barplot(df)
	plot_spending_per_customer(df)

if __name__ == "__main__":
	main()