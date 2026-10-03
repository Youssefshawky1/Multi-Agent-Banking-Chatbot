# ==== BEGIN AI-GENERATED (Claude, Anthropic) ====
import torch
from langchain_core.runnables import RunnableBranch, RunnableLambda

from training.model import IntentClassifier
from agents.agents import build_agents
from agents.llm import get_llm, SIMULATE_RATE_LIMIT, MODEL_NAMES

MAX_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.6

checkpoint = torch.load("training/classifier.pt", map_location="cpu")
vocab = checkpoint["vocab"]
id_to_label = {i: name for name, i in checkpoint["label_map"].items()}

model = IntentClassifier(vocab_size=len(vocab))
model.load_state_dict(checkpoint["model_state"])
model.eval()


def encode(text):
    tokens = text.lower().split(" ")
    encoded = [vocab.get(t, vocab["<unk>"]) for t in tokens][:MAX_LENGTH]
    return encoded + [vocab["<pad>"]] * (MAX_LENGTH - len(encoded))


def classify(text):
    x = torch.tensor([encode(text)], dtype=torch.long)
    lengths = (x != 0).sum(dim=1).clamp(min=1)
    with torch.no_grad():
        probs = torch.softmax(model(x, lengths), dim=1)[0]
    confidence, index = probs.max(dim=0)
    all_probs = {id_to_label[i]: probs[i].item() for i in range(len(probs))}
    return id_to_label[index.item()], confidence.item(), all_probs


agents_cache = {}
USAGE = {"groq": 0, "ollama": 0}


def get_agents(provider):
    if provider not in agents_cache:
        agents_cache[provider] = build_agents(get_llm(provider))
    return agents_cache[provider]


def run_agent(name, inputs):
    errors = []
    for provider in ["groq", "ollama"]:
        try:
            if provider == "groq" and SIMULATE_RATE_LIMIT:
                raise Exception(
                    "Simulated Groq rate limit (429 Too Many Requests)")
            result = get_agents(provider)[name].invoke(
                {"input": inputs["input"],
                    "chat_history": inputs["chat_history"]}
            )
            USAGE[provider] += 1
            tool_calls = [{"tool": action.tool, "input": action.tool_input, "output": str(output)[:300]}
                          for action, output in result["intermediate_steps"]]
            return {**inputs, "reply": result["output"], "provider": provider,
                    "model": MODEL_NAMES[provider], "tool_calls": tool_calls, "errors": errors}
        except Exception as e:
            errors.append(f"{provider}: {e}")
    return {**inputs, "reply": "Sorry, the assistant is busy right now. Please try again in a minute.",
            "provider": None, "model": None, "tool_calls": [], "errors": errors}


def classify_step(inputs):
    intent, confidence, probs = classify(inputs["input"])
    agent = intent if confidence >= CONFIDENCE_THRESHOLD else "general"
    return {**inputs, "intent": intent, "confidence": confidence, "probs": probs, "agent": agent}


router_chain = RunnableLambda(classify_step) | RunnableBranch(
    (lambda x: x["agent"] == "billing", RunnableLambda(
        lambda x: run_agent("billing", x))),
    (lambda x: x["agent"] == "technical", RunnableLambda(
        lambda x: run_agent("technical", x))),
    RunnableLambda(lambda x: run_agent("general", x)),
)


def route_and_run(user_message, memory):
    result = router_chain.invoke(
        {"input": user_message, "chat_history": memory.as_messages()})
    memory.add_turn(user_message, result["reply"])
    return result


if __name__ == "__main__":
    from agents.memory import ConversationMemory
    memory = ConversationMemory()
    while True:
        message = input("\nYou: ").strip()
        if message in ["", "exit"]:
            break
        r = route_and_run(message, memory)
        print(f"[intent={r['intent']} confidence={r['confidence']:.2f} agent={r['agent']} "
              f"model={r['model']} tools={[t['tool'] for t in r['tool_calls']]}]")
        print("Bot:", r["reply"])
        if r["errors"]:
            print("Errors:", r["errors"])
# ==== END AI-GENERATED ====
