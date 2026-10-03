
import streamlit as st

from agents.router import route_and_run, USAGE, CONFIDENCE_THRESHOLD
from agents.memory import ConversationMemory

DAILY_CAP = 1000
COLORS = {"billing": "#F59E0B", "technical": "#6366F1", "general": "#14B8A6"}
ICONS = {"billing": "💳", "technical": "🛠️", "general": "💬"}
PROVIDERS = {"groq": "⚡ Groq",
             "ollama": "🦙 Ollama (local)", None: "⛔ Unavailable"}

st.set_page_config(page_title="Support Bot", page_icon="💬", layout="centered",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
html {font-size: 20px;}
.block-container {max-width: 1000px;}
[data-testid="stSidebar"] {min-width: 380px; max-width: 380px;}
[data-testid="stChatMessage"] p {font-size: 1.05rem;}
.intent {font-size: 2.2rem; font-weight: 700; margin: -6px 0 8px;}
.bar {height: 12px; border-radius: 99px; background: rgba(128,128,128,.25); overflow: hidden; margin: 2px 0 14px;}
.bar div {height: 100%; border-radius: 99px;}
.tag {display: inline-block; padding: 3px 12px; border-radius: 99px; font-size: .8rem;
      font-weight: 600; margin: 6px 6px 0 0;}
</style>
""", unsafe_allow_html=True)


def bar(value, color):
    return f'<div class="bar"><div style="width:{value * 100:.1f}%;background:{color}"></div></div>'


def tag(text, color):
    return f'<span class="tag" style="background:{color}26;color:{color}">{text}</span>'


def tags_for(r):
    html = tag(f"{ICONS[r['agent']]} {r['agent']} agent", COLORS[r["agent"]])
    for call in r["tool_calls"]:
        html += tag(f"🔧 {call['tool']}", "#8B5CF6")
    return html


st.title("💬 Multi-Agent Support Bot")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.memory = ConversationMemory()
    st.session_state.last = None

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn.get("meta"):
            st.markdown(tags_for(turn["meta"]), unsafe_allow_html=True)

if user_msg := st.chat_input("How can I help?"):
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = route_and_run(user_msg, st.session_state.memory)
        st.write(result["reply"])
        st.markdown(tags_for(result), unsafe_allow_html=True)
    st.session_state.history += [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": result["reply"], "meta": result},
    ]
    st.session_state.last = result

with st.sidebar:
    st.subheader("Routing")
    r = st.session_state.last
    if r is None:
        st.caption("Send a message to see the routing decision.")
    else:
        color = COLORS[r["intent"]]
        st.caption("Predicted intent")
        st.markdown(f'<p class="intent" style="color:{color}">{ICONS[r["intent"]]} {r["intent"]}</p>',
                    unsafe_allow_html=True)
        st.caption(f"Confidence: **{r['confidence']:.1%}**")
        st.markdown(bar(r["confidence"], color), unsafe_allow_html=True)

        if r["agent"] != r["intent"]:
            st.warning(
                f"Confidence below {CONFIDENCE_THRESHOLD:.0%} → sent to general agent")

        with st.expander("All intent probabilities"):
            for name, p in sorted(r["probs"].items(), key=lambda x: -x[1]):
                st.caption(f"{ICONS[name]} {name}: {p:.1%}")
                st.markdown(bar(p, COLORS[name]), unsafe_allow_html=True)

        st.write(f"Handled by: **{r['agent']} agent**")
        st.write(f"LLM: **{PROVIDERS[r['provider']]}**")
        st.caption(f"Model: {r['model'] or 'none'}")
        for e in r["errors"]:
            st.error(e[:160])

        if r["tool_calls"]:
            st.write("Tools called: " +
                     ", ".join(f"**{c['tool']}**" for c in r["tool_calls"]))
            for call in r["tool_calls"]:
                with st.expander(f"🔧 {call['tool']} details"):
                    st.caption(f"Input: {call['input']}")
                    st.caption(f"Output: {call['output']}")
        else:
            st.write("Tools called: none")

    st.divider()
    st.subheader("Usage")
    st.write(f"{USAGE['groq']} / {DAILY_CAP} requests used")
    st.markdown(bar(min(USAGE["groq"] / DAILY_CAP, 1),
                "#14B8A6"), unsafe_allow_html=True)
    st.caption(f"Ollama (local): {USAGE['ollama']} requests")
    if st.button("Clear chat", width="stretch"):
        st.session_state.clear()
        st.rerun()
# ==== END AI-GENERATED ====
