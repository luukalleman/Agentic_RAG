# table_manager.py

from app.data.models.models import TableDescription


class TableManager:
    def __init__(self, db_handler, client):
        self.db_handler = db_handler
        self.client = client

    def create_table(self, table_name, columns, raw_data=None):
        if not self.db_handler.table_exists(table_name):
            self.db_handler.create_table(table_name, columns)
            description = self._generate_table_description(raw_data)
            print(description)
            self.db_handler.insert_row('data_sources', {
                "table_name": table_name,
                "description": description
            })

    def _generate_table_description(self, raw_data):
        if not raw_data:
            return "No description available."
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are an assistant that creates short, concise descriptions for database tables."},
                    {"role": "user", "content": f"Summarize the following data for a database table:\n\n{raw_data}"}
                ],
                response_format=TableDescription,
            )
            return completion.choices[0].message.parsed.description
        except Exception as e:
            print(f"Error generating table description: {e}")
            return "Description generation failed."
