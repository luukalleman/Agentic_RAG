# agent_core.py

import json
import logging
from openai import OpenAI


class Agent:
    def __init__(self, instructions, model="gpt-4o", functions=None,temperature=0.0, agent_id = None):
        self.client = OpenAI()
        self.functions = {}
        self.tools = []
        self.model = model
        self.instructions = instructions
        self.vector_store = None
        self.temperature = temperature
        self.agent_id = agent_id
        # Configure logging
        self.logger = logging.getLogger(__name__)
        # Register functions
        if functions:
            for function in functions:
                self.add_function(function)

        # Create the assistant
        self.assistant = self.client.beta.assistants.create(
            instructions=self.instructions,
            model=self.model,
            tools=self.tools
        )
        self.logger.info("Assistant created with model %s.", self.model)

        self.thread = None  # Conversation thread

    def add_function(self, function):
        """Register a function as a tool for the assistant."""
        func_metadata = {
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": function.parameters,
                "strict": True
            }
        }

        # Add the function to the assistant's tools
        self.tools.append(func_metadata)
        # Register the function for execution
        self.functions[function.name] = function
        self.logger.info("Registered function: %s", function.name)

    def start_conversation(self):
        """Create a new conversation thread."""
        self.thread = self.client.beta.threads.create()
        self.logger.info("Conversation thread started with ID: %s", self.thread.id)

    def send_message(self, content):
        """Send a message to the assistant and handle the response."""
        self.content = content
        if not self.thread:
            self.start_conversation()
        self.logger.debug("User: %s", content)

        self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=content
        )
        self._process_run()

    def _process_run(self):
        """Initiate a run and handle required actions."""
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            temperature=self.temperature
        )
        print(f"Run status: {run.status}")
        while run.status != 'completed':
            if run.status == 'requires_action':
                required_action = run.required_action
                print("required_action: ", required_action)
                if required_action.type == 'submit_tool_outputs':
                    tool_outputs = self._handle_function_calls(
                        required_action.submit_tool_outputs.tool_calls)
                    print("tool_outputs: ", tool_outputs)
                    run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=self.thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                    print("run: ", run)
                else:
                    self.logger.warning("Unknown required action: %s", required_action.type)
                    break
            else:
                self.logger.warning("Run status: %s", run.status)
                break

    def _handle_function_calls(self, tool_calls):
        """Execute the functions requested by the assistant."""
        tool_outputs = []
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            print(f"Running Function: {func_name}")
            if func_name in self.functions:
                args = json.loads(tool_call.function.arguments)
                # Pass context to the function
                context = {'thread_id': self.thread.id, 'question': self.content, 'agent_id': self.agent_id}
                try:
                    result = self.functions[func_name].execute(args=args, context=context)
                except Exception as e:
                    self.logger.error("Error executing function '%s': %s", func_name, str(e))
                    result = f"Error executing function '{func_name}': {str(e)}"
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": result
                })
            else:
                self.logger.error("Function '%s' not found.", func_name)
                tool_outputs.append({
                    "tool_call_id": tool_call.id,
                    "output": f"Function '{func_name}' not found."
                })
        return tool_outputs

    def get_messages(self):
        """Retrieve conversation messages."""
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id
        )
        processed_messages = []
        for message in messages:
            text_content = ''
            for content_block in message.content:
                if content_block.type == 'text':
                    text_content += content_block.text.value
            processed_messages.append({
                'role': message.role,
                'content': text_content
            })
        return processed_messages

    def get_last_response(self):
        """Retrieve the assistant's last response."""
        messages = self.get_messages()
        for message in messages:
            if message['role'] == 'assistant':
                print(message['content'])
                return message['content']
        return None