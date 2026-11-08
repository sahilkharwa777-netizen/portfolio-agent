"""Builds the SQLite database from database_schema.sql + the CSV files in data/.

Everything lives inside this folder, so the project works wherever it is unzipped.
"""
import os
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
SCHEMA_PATH = os.path.join(ROOT, "database_schema.sql")

# Parents before children because of foreign keys
TABLES = ["sectors", "securities", "benchmarks", "portfolios", "holdings",
          "transactions", "historical_prices", "portfolio_performance", "risk_metrics"]


def build_database(path: str = ":memory:", read_only: bool = True) -> sqlite3.Connection:
    """Create the schema, load all CSVs, and (by default) lock the connection to read-only.

    `PRAGMA query_only` is a second safety layer: even if the SQL validator were
    bypassed, SQLite itself would refuse INSERT/UPDATE/DELETE/DROP.
    """
    if not os.path.isfile(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        conn.executescript(f.read())
    for table in TABLES:
        csv_path = os.path.join(DATA_DIR, f"{table}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        pd.read_csv(csv_path).to_sql(table, conn, if_exists="append", index=False)
    conn.commit()
    if read_only:
        conn.execute("PRAGMA query_only = ON")
    return conn


def get_schema_text() -> str:
    """CREATE TABLE statements only (no indexes) - this is the schema context sent to the LLM."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        sql = f.read()
    return "\n".join(s.strip() + ";" for s in sql.split(";")
                     if s.strip().upper().startswith("CREATE TABLE"))
