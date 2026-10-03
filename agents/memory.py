from langchain_core.messages import AIMessage, HumanMessage


class ConversationMemory:
    def __init__(self, max_turns=6):
        self.max_turns = max_turns
        self.turns = []

    def add_turn(self, user_msg, bot_msg):
        self.turns.append((user_msg, bot_msg))
        self.turns = self.turns[-self.max_turns:]

    def as_messages(self):
        messages = []
        for user_msg, bot_msg in self.turns:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=bot_msg))
        return messages
