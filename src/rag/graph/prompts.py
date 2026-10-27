"""Prompt templates for generation, strict mode, verification, and query rewriting."""

GROUNDED_GENERATION_PROMPT = """You are a precise technical assistant.
Answer the user's question based strictly on the provided context passages below.
Extract atomic claims made in your answer and cite the supporting chunk IDs.

Output your answer strictly in the following JSON format:
{{
  "answer": "Your comprehensive, direct answer.",
  "claims": ["Individual factual claim 1", "Individual factual claim 2"],
  "citations": ["chunk_id_1", "chunk_id_2"]
}}

Context passages:
{context}

Question: {question}
"""

STRICT_GENERATION_PROMPT = """You are in STRICT MODE.
Answer the question using ONLY the provided context passages.
Under NO circumstances assume, extrapolate, or introduce facts not explicitly stated in the context.
If the context does not contain sufficient information to answer the question, state: "INSUFFICIENT CONTEXT".

Output strictly in JSON format:
{{
  "answer": "Grounded answer or INSUFFICIENT CONTEXT",
  "claims": ["Claim 1", "Claim 2"],
  "citations": ["chunk_id"]
}}

Context passages:
{context}

Question: {question}
"""

CLAIM_VERIFICATION_PROMPT = """You are an impartial verifier of factual claims.
Check whether every claim made in the proposed answer is directly entailed by the provided context.

Context:
{context}

Answer:
{answer}

Claims to verify:
{claims}

Output your verification strictly in the following JSON format:
{{
  "is_grounded": true or false,
  "unsupported_claims": ["List of claims that are not supported by the context"],
  "summary": "Brief explanation of verification decision"
}}
"""

QUERY_REWRITE_PROMPT = """Rewrite the user's question into an optimal search query for a technical documentation search engine.
Expand acronyms if helpful, remove conversational pleasantries, and focus on essential technical keywords.

Original question: {question}

Output JSON format:
{{
  "rewritten_query": "Optimized search query"
}}
"""
