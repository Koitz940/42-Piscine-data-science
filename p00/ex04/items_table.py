import pandas as pd
import psycopg2
from psycopg2 import sql
from io import StringIO

def create(name, cur):
    query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            product_id INTEGER,
            category_id BIGINT,
            category_code VARCHAR(100),
            brand VARCHAR(100)
        )
    """).format(sql.Identifier(name))

    cur.execute(query)

def fill(file, cur, conn, table):

    df = pd.read_csv(file)

    df["product_id"] = pd.to_numeric(
        df["product_id"], errors="coerce"
    )

    df["category_id"] = pd.to_numeric(
        df["category_id"],
        errors="coerce"
    ).astype("Int64")

    df["category_code"] = df["category_code"].replace(
        r"^\s*$", None, regex=True
    )

    df["brand"] = df["brand"].replace(
        r"^\s*$", None, regex=True
    )

    df = df.astype(object).where(pd.notna(df), None)

    buffer = StringIO()

    df.to_csv(
        buffer,
        index=False,
        header=False,
        na_rep="\\N"
    )

    buffer.seek(0)

    copy_query = sql.SQL("""
        COPY {} (
            product_id,
            category_id,
            category_code,
            brand
        )
        FROM STDIN
        WITH CSV NULL '\\N'
    """).format(sql.Identifier(table))

    cur.copy_expert(copy_query.as_string(cur), buffer)

    conn.commit()

def main():

    conn = psycopg2.connect(
        host="localhost",
        database="piscineds",
        user="gcassi-d",
        password="mysecretpassword"
    )

    cur = conn.cursor()

    csv_file = "../../item/item.csv"
    table_name = "item"

    create(table_name, cur)
    fill(csv_file, cur, conn, table_name)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()