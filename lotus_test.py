import pandas as pd
import lotus
from lotus.models import LM

# Configure your LLM (make sure to export your API key)
lm = LM(model="gpt-4")  # Use OpenAI GPT or any supported model
lotus.settings.configure(lm=lm)

# Sample SQL-like data (you can replace this with a real SQL query)
courses_data = {
    "Course Name": ["Intro to AI", "Operating Systems", "Data Science", "Riemannian Geometry"],
    "Description": ["Learn about artificial intelligence", "Core computer science concepts", "Data-driven decision making", "Advanced mathematics for machine learning"]
}
courses_df = pd.DataFrame(courses_data)

def ask_lotus_query(user_question):
    # Perform semantic filtering with LOTUS
    result = courses_df.sem_filter(f"{user_question} applies to {courses_data['Course Name']} with descriptions: {courses_data['Description']}")
    return result

# Example usage
user_question = "Which courses are related to computer science?"
response = ask_lotus_query(user_question)
print(response)