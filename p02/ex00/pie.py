import psycopg2
import pandas as pd
import matplotlib.pyplot as plt

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
		"SELECT event_type FROM customers",
		conn
	)

	conn.close()

	print("Counting amount of each event_type...")
	counts = df["event_type"].value_counts()

	print("plotting")
	counts.plot.pie(
		autopct="%1.1f%%"
	)

	plt.ylabel("")
	plt.show()

if __name__ == "__main__":
	main()