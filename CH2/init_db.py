import sqlite3
from pathlib import Path

Path("SalesDB").mkdir(exist_ok=True)

conn = sqlite3.connect("SalesDB/sales.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    product_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    total REAL NOT NULL
)
""")

cursor.executemany("""
INSERT INTO orders (customer_name, product_name, quantity, price, total)
VALUES (?, ?, ?, ?, ?)
""", [
    ("John Doe", "Laptop", 1, 1000.00, 1000.00),
    ("Jane Smith", "Smartphone", 2, 500.00, 1000.00),
    ("Bob Johnson", "Tablet", 3, 200.00, 600.00)
])

conn.commit()

cursor.execute("SELECT * FROM orders")
print(cursor.fetchall())

conn.close()

