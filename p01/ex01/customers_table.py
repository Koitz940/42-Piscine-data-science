import pandas as pd
import psycopg2
from psycopg2 import sql
from io import StringIO
import os

def create_date(name, cur):
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


def fill_date(file, cur, conn, table):

    df = pd.read_csv(file)

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


def create_date_tables(conn, cur):
    directory = "../customer"

    for file in sorted(os.listdir(directory)):

        if file.endswith(".csv"):
            csv_file = os.path.join(directory, file)
            table_name = file[:-4]

            cur.execute(
                "SELECT to_regclass(%s) IS NOT NULL;",
                (f"public.{table_name}",)
            )

            exists = cur.fetchone()[0]

            if exists:
                continue

            print(f"Importing {file} -> {table_name}")

            create_date(table_name, cur)
            fill_date(csv_file, cur, conn, table_name)

def create_item(name, cur):
    query = sql.SQL("""
        CREATE TABLE IF NOT EXISTS {} (
            product_id INTEGER,
            category_id BIGINT,
            category_code VARCHAR(100),
            brand VARCHAR(100)
        )
    """).format(sql.Identifier(name))

    cur.execute(query)

def fill_item(file, cur, conn, table):

    df = pd.read_csv(file)

    df["product_id"] = pd.to_numeric(
        df["product_id"], errors="coerce"
    )

    df["category_id"] = pd.to_numeric(
        df["category_id"], errors="coerce"
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

def create_item_table(conn, cur):
    csv_file = "../item/item.csv"
    table_name = "items"

    cur.execute("""
        SELECT to_regclass(%s) IS NOT NULL;
    """, (f"public.{table_name}",))

    exists = cur.fetchone()[0]

    if exists:
        return

    create_item(table_name, cur)
    fill_item(csv_file, cur, conn, table_name)


def join_date_tables(conn, cur):
    cur.execute("""
        SELECT to_regclass('public.customers') IS NOT NULL;
    """)

    if cur.fetchone()[0]:
        return

    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name LIKE 'data\\_202%%\\_%%' ESCAPE '\\';
    """)

    tables = cur.fetchall()

    union_query = " UNION ALL ".join(
        f'SELECT * FROM public."{table[0]}"'
        for table in tables
    )

    cur.execute(f"""
        CREATE TABLE customers AS
        {union_query};
    """)

    conn.commit()

def join_date_tables_clean(conn, cur):
    cur.execute("""
        SELECT to_regclass('public.customers') IS NOT NULL;
    """)

    if cur.fetchone()[0]:
        return

    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_name LIKE 'data\\_202%%\\_%%' ESCAPE '\\';
    """)

    tables = cur.fetchall()

    union_query = " UNION ".join(
        f'SELECT * FROM public."{table[0]}"'
        for table in tables
    )

    cur.execute(f"""
        CREATE TABLE customers AS
        {union_query};
    """)

    conn.commit()

def create_customer():
    conn = psycopg2.connect(
        host="localhost",
        database="piscineds",
        user="gcassi-d",
        password="mysecretpassword"
    )

    cur = conn.cursor()
    create_date_tables(conn, cur)
    create_item_table(conn, cur)
    join_date_tables(conn, cur)

    cur.close()
    conn.close()

if __name__ == "__main__":
    create_customer()