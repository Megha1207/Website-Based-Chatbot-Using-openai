import re
import ollama
from embedding_pipeline.src.retriever import Retriever
from embedding_pipeline.src.llm_client import LLMClient

MAX_DISTANCE = 0.6
MAX_CONTEXT_CHUNKS = 4

FALLBACK_MESSAGE = "The answer is not available on the provided website."

STRICT_SYSTEM_PROMPT = """You are an AI assistant answering questions strictly from the provided website content.

Rules:
- Use ONLY the information present in the context.
- Do NOT use external or general knowledge.
- Do NOT guess or infer.
- If the answer is not explicitly present, respond exactly:
  "The answer is not available on the provided website."
- Be concise and factual.
- If a list is asked, return only the list items.
"""


class QAEngine:
    def __init__(self, website_url: str, llm_provider="openai"):
        self.retriever = Retriever(website_url)
        self.llm = LLMClient(llm_provider)


    # --------------------------------------------------
    # Detect list questions
    # --------------------------------------------------
    def _is_list_question(self, question: str) -> bool:
        q = question.lower().strip()
        return q.startswith(("list ", "list all ", "what are ", "which "))

    # --------------------------------------------------
    # Extract list items generically (no hardcoding)
    # --------------------------------------------------
    def _extract_list_items(self, context: str):
        items = []
        for line in context.splitlines():
            line = line.strip()
            if re.match(r"^[-•*]\s+", line) or re.match(r"^\d+\.\s+", line):
                items.append(line)
        return list(dict.fromkeys(items))

    # --------------------------------------------------
    # Check question is about website topic
    # --------------------------------------------------
    def _topic_overlap(self, question: str, context: str) -> bool:
        q_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", question.lower()))
        c_tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", context.lower()))

        stop = {
            "what", "which", "where", "when", "that", "this",
            "with", "from", "about", "they", "their", "there"
        }

        q_tokens -= stop
        c_tokens -= stop

        return len(q_tokens & c_tokens) >= 2

    # --------------------------------------------------
    # Main QA method
    # --------------------------------------------------
    def answer(self, question: str, chat_history=None):

        # Expand retrieval query using conversation memory
        retrieval_query = question
        if chat_history:
            prev = [m["content"] for m in chat_history if m["role"] == "user"]
            if prev:
                retrieval_query = f"{prev[-1]} {question}"

        results = self.retriever.retrieve(retrieval_query)
        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return FALLBACK_MESSAGE

        # Select relevant chunks
        relevant_chunks = [
            doc for doc, dist in zip(documents, distances)
            if dist <= MAX_DISTANCE
        ][:MAX_CONTEXT_CHUNKS]

        if not relevant_chunks:
            relevant_chunks = documents[:MAX_CONTEXT_CHUNKS]

        context = "\n\n".join(relevant_chunks)

        # HARD STOP: Question must relate to site content
        if not self._topic_overlap(question, context):
            return FALLBACK_MESSAGE

        # Deterministic list extraction
        if self._is_list_question(question):
            extracted = self._extract_list_items(context)
            if extracted:
                return "\n".join(extracted)

        # LLM call (strict grounding)
        messages = [{"role": "system", "content": STRICT_SYSTEM_PROMPT}]

        if chat_history:
            messages.extend(chat_history[-4:])

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        })

        answer = self.llm.generate(messages)


        if not answer or FALLBACK_MESSAGE.lower() in answer.lower():
            return FALLBACK_MESSAGE

        return answer
