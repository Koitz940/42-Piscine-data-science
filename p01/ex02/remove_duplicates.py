import psycopg2

def remove_duplicates():
	conn = psycopg2.connect(
		host="localhost",
		database="piscineds",
		user="gcassi-d",
		password="mysecretpassword"
	)

	cur = conn.cursor()

	cur.execute(f"""
		CREATE TABLE customers_clean AS
		SELECT event_time, event_type, product_id, price, user_id, user_session
		FROM (
			SELECT
				event_time,
				event_type,
				product_id,
				price,
				user_id,
				user_session,
				event_time - LAG(event_time) OVER (
					PARTITION BY
						event_type,
						product_id,
						price,
						user_id,
						user_session
					ORDER BY event_time
				) AS time_gap
			FROM customers
		) AS ordered
		WHERE time_gap IS NULL
		OR time_gap > INTERVAL '1 second';
	""")
	cur.execute("DROP TABLE customers;")
	cur.execute("ALTER TABLE customers_clean RENAME TO customers;")


	conn.commit()

	cur.close()
	conn.close()

if __name__ == "__main__":
    remove_duplicates()