"""
mcp_multitool.py
Example MCP server exposing three tools:
  - calc: basic math
  - query_sqlite: simple SQLite queries
  - get_weather: fetch quick weather info
"""

import os, math, sqlite3, requests
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("example-multitool-server")

DB_PATH = os.path.abspath("demo.db")

# --- Setup demo SQLite DB ---
def init_demo_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            city TEXT NOT NULL,
            spend REAL NOT NULL
        )
    """)
    if cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
        cur.executemany(
            "INSERT INTO customers (name, city, spend) VALUES (?, ?, ?)",
            [
                ("Akira", "Tokyo", 1200.50),
                ("Mika", "Kyoto", 830.00),
                ("Ken",  "Osaka", 1520.75),
                ("Aoi",  "Kyoto", 410.10),
            ],
        )
    con.commit()
    con.close()

init_demo_db()

# --- Tool 1: Calculator ---
@mcp.tool()
def calc(op: str, a: float, b: float = None) -> float:
    """Perform a math operation: add, sub, mul, div, pow, sqrt."""
    if op == "sqrt":
        if a < 0:
            raise ValueError("sqrt expects non-negative 'a'")
        return math.sqrt(a)
    if b is None:
        raise ValueError("This operation requires parameter 'b'")
    if op == "add": return a + b
    if op == "sub": return a - b
    if op == "mul": return a * b
    if op == "div":
        if b == 0: raise ValueError("Division by zero")
        return a / b
    if op == "pow": return math.pow(a, b)
    raise ValueError(f"Unsupported op: {op}")

# --- Tool 2: SQLite query ---
@mcp.tool()
def query_sqlite(sql: str, city: str = None, limit: int = 10, name_like: str = None):
    """
    Query demo SQLite database.
    sql = "by_city" | "top_spenders" | "by_name_like"
    """
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    if sql == "by_city" and city:
        cur.execute("SELECT id,name,city,spend FROM customers WHERE city=?",(city,))
    elif sql == "top_spenders":
        cur.execute("SELECT id,name,city,spend FROM customers ORDER BY spend DESC LIMIT ?",(limit,))
    elif sql == "by_name_like" and name_like:
        cur.execute("SELECT id,name,city,spend FROM customers WHERE name LIKE ?",(name_like,))
    else:
        raise ValueError("Invalid query or missing params")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

# --- Tool 3: Weather ---
@mcp.tool()
def get_weather(city: str) -> str:
    """Fetch quick weather summary for a city."""
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        return resp.text if resp.ok else f"Failed: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"

# --- Entry point ---
if __name__ == "__main__":
    mcp.run()   # blocking server loop
