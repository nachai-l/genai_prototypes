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
        if city is None:        
            cur.execute("SELECT id,name,city,spend FROM customers ORDER BY spend DESC LIMIT ?",(limit,))
        else:
            cur.execute(
                "SELECT id,name,city,spend FROM customers WHERE city = ? ORDER BY spend DESC LIMIT ?",
                (city, limit)
            )
    elif sql == "by_name_like" and name_like:
        cur.execute("SELECT id,name,city,spend FROM customers WHERE name LIKE ?",(name_like,))
    else:
        raise ValueError("Invalid query or missing params")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

# --- Tool 2.5: SQLite query flexible ---
@mcp.tool()
def query_sqlite_flexible(
    table: str = "customers",
    select: dict | list | None = None,
    where: list[dict] | None = None,
    group_by: list[str] | None = None,
    having: list[dict] | None = None,
    order_by: list[dict] | None = None,
    limit: int | None = 50,
    offset: int | None = None,
):
    """
    Flexible, parameterized SQL builder for SQLite.

    Arguments:
      table: target table name (e.g., "customers")
      select:
        - list[str]: plain columns, e.g. ["id","name","city"]
        - dict: can be either:
            {"columns":[...], "aggs":{alias:{"fn":"SUM","expr":"spend"}, ...}}
            OR
            {alias:{"fn":"SUM","expr":"spend"}, ...}
      where / having: list of {"left":<col_or_alias>, "op":<str>, "right":<val|list>}
      group_by: list[str]
      order_by: list of {"expr":"<col_or_alias>","dir":"ASC|DESC"}
      limit / offset: pagination

    Notes:
      - All values are safely bound as parameters.
      - Allowed ops: =, !=, <, >, <=, >=, LIKE, IN, BETWEEN
      - Aggregates allowed: SUM, AVG, MIN, MAX, COUNT
    """
    import sqlite3

    ALLOWED_OPS = {"=", "!=", "<", ">", "<=", ">=", "LIKE", "IN", "BETWEEN"}
    ALLOWED_AGG = {"SUM","AVG","MIN","MAX","COUNT"}

    def ident(name: str) -> str:
        """Minimal identifier guard (safe against SQL injection)."""
        if not isinstance(name, str) or not name.replace("_","").isalnum():
            raise ValueError(f"Invalid identifier: {name!r}")
        return name

    def build_condition(cond: dict, params: list) -> str:
        """Translate a {left,op,right} dict into SQL + params."""
        left, op, right = cond.get("left"), cond.get("op","=").upper(), cond.get("right")
        if not left:
            raise ValueError("Condition missing 'left'")
        if op not in ALLOWED_OPS:
            raise ValueError(f"Unsupported op: {op}")
        left_sql = ident(left)

        if op == "IN":
            if not isinstance(right, (list, tuple)) or not right:
                raise ValueError("IN expects non-empty list/tuple")
            qmarks = ",".join(["?"] * len(right))
            params.extend(right)
            return f"{left_sql} IN ({qmarks})"
        elif op == "BETWEEN":
            if not (isinstance(right,(list,tuple)) and len(right)==2):
                raise ValueError("BETWEEN expects a 2-element list/tuple")
            params.extend([right[0], right[1]])
            return f"{left_sql} BETWEEN ? AND ?"
        else:
            params.append(right)
            return f"{left_sql} {op} ?"

    # --- SELECT ---
    columns_sql, params = [], []
    if select is None:
        columns_sql.append("*")
    elif isinstance(select, list):
        columns_sql += [ident(c) for c in select]
    elif isinstance(select, dict):
        if "columns" in select or "aggs" in select:
            for c in select.get("columns", []):
                columns_sql.append(ident(c))
            for alias, spec in (select.get("aggs") or {}).items():
                fn = spec.get("fn","").upper()
                expr = ident(spec.get("expr",""))
                if fn not in ALLOWED_AGG:
                    raise ValueError(f"Unsupported aggregate: {fn}")
                columns_sql.append(f"{fn}({expr}) AS {ident(alias)}")
        else:
            for alias, spec in select.items():
                fn = spec.get("fn","").upper()
                expr = ident(spec.get("expr",""))
                if fn not in ALLOWED_AGG:
                    raise ValueError(f"Unsupported aggregate: {fn}")
                columns_sql.append(f"{fn}({expr}) AS {ident(alias)}")
    else:
        raise ValueError("Invalid 'select' type")

    sql_parts = [f"SELECT {', '.join(columns_sql)} FROM {ident(table)}"]

    # --- WHERE ---
    if where:
        where_clauses = [build_condition(c, params) for c in where]
        sql_parts.append("WHERE " + " AND ".join(where_clauses))

    # --- GROUP BY ---
    if group_by:
        sql_parts.append("GROUP BY " + ", ".join(ident(c) for c in group_by))

    # --- HAVING ---
    if having:
        having_clauses = [build_condition(c, params) for c in having]
        sql_parts.append("HAVING " + " AND ".join(having_clauses))

    # --- ORDER BY ---
    if order_by:
        ords = []
        for ob in order_by:
            expr = ident(ob.get("expr",""))
            direction = ob.get("dir","ASC").upper()
            if direction not in {"ASC","DESC"}:
                raise ValueError("ORDER BY dir must be ASC or DESC")
            ords.append(f"{expr} {direction}")
        sql_parts.append("ORDER BY " + ", ".join(ords))

    # --- LIMIT/OFFSET ---
    if limit is not None:
        if not isinstance(limit,int) or limit <= 0:
            raise ValueError("limit must be positive int")
        sql_parts.append("LIMIT ?")
        params.append(limit)
    if offset is not None:
        if not isinstance(offset,int) or offset < 0:
            raise ValueError("offset must be non-negative int")
        sql_parts.append("OFFSET ?")
        params.append(offset)

    sql = " ".join(sql_parts)

    # --- Execute ---
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    try:
        cur.execute(sql, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        con.close()

    return {"sql": sql, "params": params, "rows": rows}

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
