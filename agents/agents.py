from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.tools import check_invoice, search_docs

DOMAIN = "Digital Banking Support"

SYSTEM_PROMPTS = {
    "billing": (
        f"You are the billing support agent for {DOMAIN}. You handle balances, charges, fees, "
        "refunds, transfers and top-ups. Whenever the user mentions a customer ID, call "
        "check_invoice for that exact ID, even if you already looked up a different ID earlier "
        "in the conversation. Answer only from the result of that latest call and never reuse "
        "transactions or amounts from another customer. If the user says they were charged twice "
        "but the tool result shows no duplicate transaction, say you cannot see a duplicate and "
        "list what the tool does show. If no customer ID is given, ask for it (format C001). "
        "You cannot issue refunds yourself; say a review request has been noted. Do not promise "
        "timelines, notifications or statement descriptions. Keep answers short and never invent account data."
    ),
    "technical": (
        f"You are the technical support agent for {DOMAIN}. You handle cards not working, PIN, "
        "passcode, app login, contactless and security problems. Always call search_docs and base "
        "your answer on it. If the docs do not cover the question, say so honestly."
    ),
    "general": (
        f"You are the general support agent for {DOMAIN}. You handle greetings, card delivery, "
        "supported countries and currencies, age limits and anything else. Use search_docs for "
        "questions about the bank, otherwise answer briefly and directly."
    ),
}

AGENT_TOOLS = {
    "billing": [check_invoice],
    "technical": [search_docs],
    "general": [search_docs],
}


def build_agent(llm, system_prompt, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True, max_iterations=4)


def build_agents(llm):
    return {name: build_agent(llm, SYSTEM_PROMPTS[name], AGENT_TOOLS[name]) for name in SYSTEM_PROMPTS}
