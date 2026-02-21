import os
import httpx
from openai import OpenAI
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You run in a loop of Thought, Action, Observation.
At the end of the loop, output Final Answer.
Use these tools:
  - calculate: evaluates math expressions (e.g. '2 + 2')
  - web_search: searches DuckDuckGo for info
"""

def web_search(query):
    # Simple DuckDuckGo HTML search for demo purposes
    resp = httpx.get(f"https://html.duckduckgo.com/html/?q={query}")
    return resp.text[:500] if resp.status_code == 200 else "Search failed"

def calculate(expr):
    try:
        # Warning: eval is used for demo simplicity. Protect in prod.
        return str(eval(expr, {"__builtins__": None}))
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "web_search": web_search,
    "calculate": calculate
}

def parse_action(response):
    lines = response.split('\n')
    for line in lines:
        if line.startswith("Action:"):
            # Expected format: Action: tool_name('arg')
            try:
                action_str = line.replace("Action:", "").strip()
                func_name = action_str.split('(')[0]
                arg = action_str.split("('")[1].split("')")[0]
                return func_name, arg
            except:
                pass
    return None, None

class ReActAgent:
    def __init__(self, system_prompt):
        self.messages = [{"role": "system", "content": system_prompt}]

    def run(self, max_turns=5):
        for _ in range(max_turns):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=self.messages
            ).choices[0].message.content
            
            print(f"\n{response}")
            self.messages.append({"role": "assistant", "content": response})

            action, arg = parse_action(response)
            if not action:
                return response # Final Answer reached

            print(f"Executing: {action}('{arg}')")
            observation = TOOLS[action](arg)
            print(f"Observation: {observation}")
            
            self.messages.append({"role": "user", "content": f"Observation: {observation}"})
        
        return "Max turns reached."

if __name__ == "__main__":
    agent = ReActAgent(SYSTEM_PROMPT)
    agent.messages.append({"role": "user", "content": "What is the population of Tokyo divided by the population of Iceland?"})
    agent.run()
