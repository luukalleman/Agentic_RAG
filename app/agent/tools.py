# function.py

import json
import os
from app.data.input.orders import df_orders
from app.data.handlers.db_handler import DatabaseHandler
from app.data.handlers.embedding_handler import EmbeddingHandler
from app.config.config import get_db_config
import os
from app.rag.rag import RAGPipeline

class Function:
    def __init__(self, func, name, description, parameters):
        self.func = func  # The actual Python function
        self.name = name
        self.description = description
        self.parameters = parameters

    def execute(self, args, context):
        """Execute the encapsulated function with provided arguments and context."""
        return self.func(args=args, context=context)

def get_order_status(args, context):
    """
    Retrieve the current status of an order given its order number.
    """
    order_number = args.get('order_number')
    if not order_number:
        return "Order number is missing."

    order_number = str(order_number)
    status = df_orders.loc[df_orders['order_number'] == order_number, 'status']
    if not status.empty:
        return f"The status of order number {order_number} is {status.values[0]}."
    else:
        return f"Order number {order_number} not found."

def get_estimated_delivery_date(args, context):
    """
    Provide the estimated delivery date for an order given its order number.
    """
    order_number = args.get('order_number')
    if not order_number:
        return "Order number is missing."

    order_number = str(order_number)
    delivery_date = df_orders.loc[df_orders['order_number'] == order_number, 'estimated_delivery']
    if not delivery_date.empty:
        return f"The estimated delivery date for order number {order_number} is {delivery_date.values[0]}."
    else:
        return f"Order number {order_number} not found."

def escalate_to_human(args, context):
    """
    Escalate the conversation to a human by saving the thread ID, reason, and contact info to a JSON file.
    Ensures the directory and file exist, and creates them if necessary.
    """
    reason = args.get('reason')
    contact_info = args.get('contact_info')
    thread_id = context.get('thread_id')

    if not reason or not contact_info:
        return "Reason and contact information are required to escalate."

    # Define the absolute file path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '../data/output/escalations.json')
    directory = os.path.dirname(file_path)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Initialize escalations list
    escalations = []

    # Check if the file exists and load existing data
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                escalations = json.load(f)
                if not isinstance(escalations, list):
                    raise ValueError("Invalid data format in JSON file")
        except (json.JSONDecodeError, ValueError):
            # Handle invalid JSON or format by resetting to an empty list
            escalations = []

    # Create the escalation entry
    escalation_data = {
        'thread_id': thread_id,
        'reason': reason,
        'contact_info': contact_info
    }

    # Append the new escalation
    escalations.append(escalation_data)

    # Write the updated list back to the file
    with open(file_path, 'w') as f:
        json.dump(escalations, f, indent=4)

    return "Thank you. I've escalated your request to a human representative, and they will contact you shortly."


def look_up_data(args, context):
    """
    Retrieve the current status of an order given its order number.
    """
    question = context.get('question')
    agent_id = context.get('agent_id')
    print(agent_id)
    # Initialize Database and Embedding handlers
    db_handler = DatabaseHandler(**get_db_config())
    embedding_handler = EmbeddingHandler(
        model_name="openai",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(db_handler, embedding_handler)
    answers = rag_pipeline.retrieve(query=question, agent_id= agent_id)

    # Format the answers as a structured string
    formatted_answers = "\n".join([
        f"{i+1}. \"{answer['answer']}\" (Similarity: {answer['similarity']:.3f})"
        for i, answer in enumerate(answers)
    ])
    print(formatted_answers)
    return formatted_answers
# ---------------------------------------------------
# Create Function instances
# ---------------------------------------------------

get_order_status_function = Function(
    func=get_order_status,
    name="get_order_status",
    description="Retrieve the current status of an order given its order number.",
    parameters={
        "type": "object",
        "properties": {
            "order_number": {
                "type": "string",
                "description": "The unique order number assigned to the customer's order."
            }
        },
        "required": ["order_number"],
        "additionalProperties": False
    }
)

get_estimated_delivery_date_function = Function(
    func=get_estimated_delivery_date,
    name="get_estimated_delivery_date",
    description="Provide the estimated delivery date for an order given its order number.",
    parameters={
        "type": "object",
        "properties": {
            "order_number": {
                "type": "string",
                "description": "The unique order number assigned to the customer's order."
            }
        },
        "required": ["order_number"],
        "additionalProperties": False
    }
)

escalate_to_human_function = Function(
    func=escalate_to_human,
    name="escalate_to_human",
    description="Escalate the conversation to a human representative when the assistant cannot assist the user.",
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "The reason why the conversation should be escalated to a human."
            },
            "contact_info": {
                "type": "string",
                "description": "The contact information of the person that wants to speak with a human."
            }
        },
        "required": ["reason", "contact_info"],
        "additionalProperties": False
    }
)

look_up_data_function = Function(
    func=look_up_data,
    name="look_up_data",
    description="Look up data from the amazon knowledgebase, things like Q&A's, internal documents etc can all be found here.",
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "The source for this question. if this can be found in the Q&A, return 'qa_pairs', otherwise we should be able to find it in 'documents'"
            }
        },
        "required": ["source"],
        "additionalProperties": False
    }
)