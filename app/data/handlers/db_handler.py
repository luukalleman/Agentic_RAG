import psycopg2
from psycopg2.extras import RealDictCursor

class DatabaseHandler:
    def __init__(self, dbname, user, password, host='127.0.0.1', port=5432):
        """Initialize the database connection."""
        self.connection = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)

    def create_table(self, table_name, columns):
        """
        Create a table in the database.
        Args:
            table_name (str): Name of the table to create.
            columns (dict): Column names and their SQL types.
        """
        column_definitions = ", ".join([f"{col} {dtype}" for col, dtype in columns.items()])
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions});"
        self.cursor.execute(query)
        self.connection.commit()

    def insert_row(self, table_name, data, conflict_column=None, update_on_conflict=True):
        """
        Insert a row into a table, with optional conflict handling.
        Args:
            table_name (str): Name of the table.
            data (dict): Column names and their values.
            conflict_column (str): The column to check for conflicts (e.g., 'table_name').
            update_on_conflict (bool): Whether to update the row if a conflict occurs.
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        values = tuple(data.values())

        if conflict_column and update_on_conflict:
            update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in data.keys()])
            query = f"""
            INSERT INTO {table_name} ({columns}) 
            VALUES ({placeholders}) 
            ON CONFLICT ({conflict_column}) 
            DO UPDATE SET {update_clause}
            """
        elif conflict_column:
            query = f"""
            INSERT INTO {table_name} ({columns}) 
            VALUES ({placeholders}) 
            ON CONFLICT ({conflict_column}) 
            DO NOTHING
            """
        else:
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"

        try:
            self.cursor.execute(query, values)
            self.connection.commit()
        except Exception as e:
            self.connection.rollback()  # Rollback transaction in case of error
            raise Exception(f"Error inserting row into {table_name}: {str(e)}")

    def fetch_data(self, table_name, columns=None, conditions=None, limit=None):
        """
        Fetch data from a table.
        Args:
            table_name (str): Name of the table.
            columns (list, optional): List of columns to retrieve. Defaults to None (all columns).
            conditions (str, optional): SQL WHERE conditions. Defaults to None.
            limit (int, optional): Number of rows to fetch. Defaults to None.
        Returns:
            list: List of fetched rows.
        """
        if columns:
            columns_str = ", ".join(columns)
        else:
            columns_str = "*"

        query = f"SELECT {columns_str} FROM {table_name}"
        if conditions:
            query += f" WHERE {conditions}"
        if limit:
            query += f" LIMIT {limit}"
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        return rows

    def update_row(self, table_name, updates, conditions):
        """
        Update rows in a table.
        Args:
            table_name (str): Name of the table.
            updates (dict): Column names and their new values.
            conditions (str): SQL WHERE conditions.
        """
        set_clause = ", ".join([f"{col} = %s" for col in updates.keys()])
        values = tuple(updates.values())
        query = f"UPDATE {table_name} SET {set_clause} WHERE {conditions}"
        self.cursor.execute(query, values)
        self.connection.commit()

    def delete_row(self, table_name, conditions):
        """
        Delete rows from a table.
        Args:
            table_name (str): Name of the table.
            conditions (str): SQL WHERE conditions.
        """
        query = f"DELETE FROM {table_name} WHERE {conditions}"
        self.cursor.execute(query)
        self.connection.commit()

    def table_exists(self, table_name):
        """
        Check if a table exists in the database.
        Args:
            table_name (str): Name of the table to check.
        Returns:
            bool: True if the table exists, False otherwise.
        """
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = %s
        );
        """
        self.cursor.execute(query, (table_name,))
        result = self.cursor.fetchone()
        return result['exists']

    def close_connection(self):
        """Close the database connection."""
        self.cursor.close()
        self.connection.close()