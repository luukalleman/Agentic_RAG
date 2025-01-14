import requests

# Bubble Data API configuration
BASE_URL = "https://upside-ai-agent.bubbleapps.io/version-test/api/1.1/obj"
API_TOKEN = "1eddf202be914c89b9389f65c3075349"  # Replace with your API token

def fetch_ids(table_name):
    """
    Fetch only the ID values from a Bubble Data API table.
    
    Args:
        table_name (str): The name of the table to fetch data from.
    
    Returns:
        list: List of IDs retrieved from the table.
    """
    url = f"{BASE_URL}/{table_name}"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",  # Add the Bearer token for authentication
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Extract IDs from the response
            data = response.json().get("response", {}).get("results", [])
            ids = [item["_id"] for item in data]  # `_id` is the default ID field in Bubble
            return ids
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    table_name = "AGENT_config"  # Replace with the actual table name

    ids = fetch_ids(table_name)
    if ids:
        print(f"Fetched {len(ids)} IDs from the table:")
        print(ids)
    else:
        print("No data found or an error occurred.")