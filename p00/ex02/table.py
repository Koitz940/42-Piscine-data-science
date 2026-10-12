import pandas as pd
import psycopg2
from psycopg2 import sql
from io import StringIO


def create(name, cur):
    query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            event_time TIMESTAMP,
            event_type VARCHAR(50),
            product_id INTEGER,
            price NUMERIC(10, 2),
            user_id BIGINT,
            user_session UUID
        )
    """).format(sql.Identifier(name))

    cur.execute(query)


def fill(file, cur, conn, table):

    df = pd.read_csv(file)

    # Clean data
    df["event_time"] = pd.to_datetime(
        df["event_time"].str.replace(" UTC", "", regex=False),
        errors="coerce"
    )

    df["product_id"] = pd.to_numeric(
        df["product_id"], errors="coerce"
    )

    df["price"] = pd.to_numeric(
        df["price"], errors="coerce"
    )

    df["user_id"] = pd.to_numeric(
        df["user_id"], errors="coerce"
    )

    df["user_session"] = df["user_session"].where(
        df["user_session"].str.match(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}$",
            na=False
        )
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
            event_time,
            event_type,
            product_id,
            price,
            user_id,
            user_session
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

    csv_file = "../customer/data_2022_dec.csv"
    table_name = "data_2022_dec"

    create(table_name, cur)
    fill(csv_file, cur, conn, table_name)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
