import psycopg2

def join():
    conn = psycopg2.connect(
        host="localhost",
        database="piscineds",
        user="gcassi-d",
        password="mysecretpassword"
    )

    cur = conn.cursor()

    cur.execute("""
        ALTER TABLE customers
        ADD COLUMN IF NOT EXISTS category_id BIGINT,
        ADD COLUMN IF NOT EXISTS category_code TEXT,
        ADD COLUMN IF NOT EXISTS brand TEXT;
    """)

    cur.execute("""
        ALTER TABLE customers
        ALTER COLUMN category_id TYPE BIGINT;
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_item_product_id
        ON items(product_id);
    """)

    conn.commit()

    cur.execute("""
		WITH merged_items AS (
			SELECT
				product_id,
				MAX(category_id) AS category_id,
				MAX(category_code) AS category_code,
				MAX(brand) AS brand
			FROM items
			GROUP BY product_id
		)
		UPDATE customers AS c
		SET
			category_id = i.category_id,
			category_code = i.category_code,
			brand = i.brand
		FROM merged_items AS i
		WHERE i.product_id = c.product_id;
	""")
    print(f"Updated {cur.rowcount:,} customers")

    conn.commit()

    cur.close()
    conn.close()


if __name__ == "__main__":
    join()