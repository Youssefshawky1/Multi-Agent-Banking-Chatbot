from pathlib import Path
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "kb"

BILLING_DB = {
    "C001": {"name": "Ahmed Ali", "balance": 12450.75, "amount_due": 3200.00, "status": "unpaid",
             "last_transactions": "Carrefour 845.20 EGP (charged twice), Uber 120.00 EGP"},
    "C002": {"name": "Sara Hassan", "balance": 980.10, "amount_due": 0.00, "status": "paid",
             "last_transactions": "Netflix 250.00 EGP"},
    "C003": {"name": "Omar Nabil", "balance": -150.00, "amount_due": 1500.00, "status": "overdue",
             "last_transactions": "Late fee 150.00 EGP"},
}


@tool
def check_invoice(customer_id: str) -> str:
    """Look up a customer's balance, amount due, invoice status and last transactions. Input: customer ID like C001."""
    record = BILLING_DB.get(customer_id.strip().upper())
    if record is None:
        return f"No account found for customer ID {customer_id}."
    return (f"Customer {customer_id.upper()} ({record['name']}): balance {record['balance']} EGP, "
            f"amount due {record['amount_due']} EGP, status {record['status']}, "
            f"last transactions: {record['last_transactions']}.")


retriever = None


def get_retriever():
    global retriever
    if retriever is None:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        docs = [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
                for p in KB_DIR.glob("*.md")]
        if not docs:
            return None
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50).split_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")
        retriever = FAISS.from_documents(
            chunks, embeddings).as_retriever(search_kwargs={"k": 3})
    return retriever


@tool
def search_docs(query: str) -> str:
    """Search the bank's help-centre knowledge base for cards, app, PIN, security, delivery, countries and policies."""
    r = get_retriever()
    if r is None:
        return "The knowledge base is empty."
    results = r.invoke(query)
    return "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in results)
