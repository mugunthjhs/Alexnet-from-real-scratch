import pandas as pd
import duckdb
import time


class ExcelSQLEngine:
    def __init__(self, excel_path):
        print("Loading Excel...")

        start = time.time()

        self.df = pd.read_excel(
            excel_path,
            engine="openpyxl"
        )

        print(f"Rows Loaded: {len(self.df):,}")
        print(f"Columns: {len(self.df.columns)}")
        print(f"Load Time: {time.time() - start:.2f} sec")

        self.con = duckdb.connect()

        self.con.register(
            "maintenance_logs",
            self.df
        )

        print("DuckDB Ready")

    def schema(self):
        return self.con.execute("""
            DESCRIBE maintenance_logs
        """).fetchdf()

    def query(self, sql):
        start = time.time()

        try:
            result = self.con.execute(sql).fetchdf()

            print(
                f"Query Time: {time.time() - start:.3f} sec"
            )

            return result

        except Exception as e:
            print(f"SQL Error: {e}")
            return None

    def close(self):
        self.con.close()


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    excel_file = r"Z:\alexnet\Alexnet-from-real-scratch\excel_to_sql\DefectDetection.xlsx"

    db = ExcelSQLEngine(excel_file)

    print("\n===== SCHEMA =====")
    print(db.schema())

    # -----------------------------------------
    # Query 1
    # -----------------------------------------
    sql = """
    SELECT
        cause,
        COUNT(*) AS total_faults
    FROM maintenance_logs
    GROUP BY cause
    ORDER BY total_faults DESC
    """

    print("\n===== FAULTS BY CAUSE =====")
    print(db.query(sql))

    # -----------------------------------------
    # Query 2
    # -----------------------------------------
    sql = """
    SELECT
        technician,
        COUNT(*) AS jobs_handled
    FROM maintenance_logs
    WHERE technician IS NOT NULL
    GROUP BY technician
    ORDER BY jobs_handled DESC
    LIMIT 10
    """

    print("\n===== TOP TECHNICIANS =====")
    print(db.query(sql))

    # -----------------------------------------
    # Query 3
    # -----------------------------------------
    sql = """
    SELECT
        line,
        COUNT(*) AS fault_count
    FROM maintenance_logs
    GROUP BY line
    ORDER BY fault_count DESC
    LIMIT 20
    """

    print("\n===== TOP LINES =====")
    print(db.query(sql))

    # -----------------------------------------
    # Dynamic Query
    # -----------------------------------------
    user_sql = """
    SELECT
        department,
        COUNT(*) AS total
    FROM maintenance_logs
    GROUP BY department
    ORDER BY total DESC
    """

    print("\n===== USER QUERY =====")
    print(db.query(user_sql))

    db.close()