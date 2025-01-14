from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from app.agent.agent import Agent
from app.agent.prompts import instructions
from app.agent.tools import get_order_status_function, look_up_data_function, get_estimated_delivery_date_function, escalate_to_human_function
from app.data.insert.document_processor import DocumentProcessor
from app.config.config import get_db_config, get_embedding_config
import os
from flask_cors import CORS

# Flask app initialization
app = Flask(__name__)
# Enable CORS
# Allow all origins for /api routes# Flask app initialization
CORS(app, resources={r"/api/*": {"origins": "*"}})

# File upload folder
UPLOAD_FOLDER = "app/data/input/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Initialize the document processor
processor = DocumentProcessor(get_db_config(), get_embedding_config())

# Agent endpoint


@app.route('/api/agent', methods=['POST'])
def agent_endpoint():
    print(request)
    print(f"Query Parameters: {request.args}")  # for GET query parameters
    print(f"Raw Body:\n{request.get_data(as_text=True)}")  # Raw body content (can be JSON or text)

    data = request.get_json()
    print(f"data: {data}")
    question = data.get("question")
    function_names = data.get("functions")
    agent_id = data.get("agent_id")
    if not question or not function_names:
        return jsonify({"error": "Missing question or functions"}), 400
    print(f"function_names: {function_names}")
    # Available functions
    available_functions = {
        "get_order_status": get_order_status_function,
        "get_estimated_delivery_date": get_estimated_delivery_date_function,
        "escalate_to_human": escalate_to_human_function,
        "look_up_data": look_up_data_function
    }
    selected_functions = [available_functions[name]
                          for name in function_names if name in available_functions]
    if not selected_functions:
        return jsonify({"error": "Invalid functions provided"}), 400
    # Initialize agent
    agent = Agent(instructions=instructions,
                  functions=selected_functions, agent_id=agent_id)

    # Send question to agent
    agent.send_message(question)
    response = agent.get_last_response()

    if response:
        return jsonify({"response": response}), 200
    else:
        return jsonify({"error": "No response from the agent"}), 500

# File upload endpoint


@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    agent_id = request.form.get("agent_id")
    source_name = request.form.get("source_name", "Uploaded Data")
    source_metadata = {"source": request.form.get(
        "source_metadata", "Custom Upload")}
    print(f"now uploading to db for agentid = {agent_id}")

    if not agent_id:
        return jsonify({"error": "Missing agent_id"}), 400

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        # Save and process the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            processor.pdf_processor.process_pdf(
                file_path,
                source_name,
                {"source": source_metadata},
                chunk_type='static',
                agent_id=agent_id
            )
            return jsonify({"message": f"File '{filename}' processed successfully for agent ID {agent_id}"}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to process the file: {str(e)}"}), 500


# Start the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
