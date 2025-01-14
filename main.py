# main.py

from app.agent.agent import Agent
from app.agent.prompts import instructions
from app.agent.tools import get_order_status_function, get_estimated_delivery_date_function, escalate_to_human_function


def main():
    agent = Agent(instructions=instructions, 
                  functions=[get_order_status_function, get_estimated_delivery_date_function, escalate_to_human_function])

    print("Hi, welcome to the ShopWise Assistant, what can I do to help you?")
    
    while True:
        user_input = input("User: ").strip()
        
        if user_input.lower() == "exit":
            print("Thanks for this conversation!")
            break
        
        agent.send_message(user_input)
        ai_response = agent.get_last_response()
        print(f"Assistant: {ai_response}")
    
if __name__ == "__main__":
    main()