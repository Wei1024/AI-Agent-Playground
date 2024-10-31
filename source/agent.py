# agent_module.py
import os
from typing import List, Dict
from tavily import TavilyClient
from autogen import AssistantAgent, UserProxyAgent, register_function, ChatResult
from autogen.agentchat.contrib.capabilities import teachability
from autogen.cache import Cache
from autogen.coding import LocalCommandLineCodeExecutor
from dotenv import load_dotenv

load_dotenv()

# Initialize TavilyClient
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def search_tool(query: str) -> str:
    return tavily.get_search_context(query=query, search_depth="advanced")

# Configurations for models
config_list = [
    {"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]},
    {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
]

# Helper function to format messages
def format_messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    ReAct_prompt = """
    Answer the following questions as best you can. You have access to tools provided.

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take
    Action Input: the input to the action
    Observation: the result of the action
    ... (this process can repeat multiple times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question, with the urls used for the answer.

    Begin!
    Question: {input}
    """
    formatted_prompt = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages)
    return ReAct_prompt.format(input=formatted_prompt)

# Code executor setup
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

# Initialize agents
user_proxy = UserProxyAgent(
    name="User",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
)

assistant = AssistantAgent(
    name="Assistant",
    system_message="""
        You are a helpful AI assistant.
        Solve tasks using tools you are provided and language skills; use your coding skill if the task cannot be solved by the tools or your language skill.

        Here are the tools you can use and the scenerios to use them:

            1. Search tool: provides search ability for you to get general web search results

        In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.

            1. When you need to collect info that cannot be acquired by the search tool, use the code to output the info you need, for example, use specific api to gain specific knowledge, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
            2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.

        Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
        When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
        If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
        If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
        When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
        Reply "TERMINATE" in the end of the final answer sentence when everything is done. If it seems there was no question or input provided and you are going on loop, also reply "TERMINATE"
    
        """,
    llm_config={"config_list": config_list, "cache_seed": None},
)

# Teachability setup for memory
teachability = teachability.Teachability(
    verbosity=0,
    reset_db=False,
    path_to_db_dir="./.tmp/teachability_db",
    recall_threshold=1.5,
)
teachability.add_to_agent(assistant)

# Register search tool
register_function(
    search_tool,
    caller=assistant,
    executor=user_proxy,
    name="search_tool",
    description="Search the web for the given query",
)

def get_agent_response(messages: List[Dict[str, str]]) -> ChatResult:
    formatted_prompt = format_messages_to_prompt(messages)
    with Cache.disk(cache_seed=43) as cache:
        return user_proxy.initiate_chat(
            assistant,
            message=lambda sender, recipient, context: formatted_prompt,
            context={"question": formatted_prompt},
            cache=cache,
        )

def parse_agent_response(chat_result: ChatResult) -> str:
    # Initialize final answer and URLs list
    final_answer = ""
    urls = []

    # Iterate over chat history
    for entry in chat_result.chat_history:
        content = entry.get("content", "")
        
        # Ensure content is not None before checking for "Final Answer"
        if content and "Final Answer" in content and "TERMINATE" in content:
            final_answer = content.split("Final Answer:")[-1].split("TERMINATE")[0].strip()
        
        # Collect URLs from search tool responses
        if entry.get("tool_responses"):
            for response in entry["tool_responses"]:
                response_content = response.get("content", "")
                urls += [url["url"] for url in eval(response_content) if "url" in url]

    # Format the final answer with URLs appended
    if urls:
        final_answer += "\n\nSources:\n" + "\n".join(urls)
    
    return final_answer
