import os
import re
from typing import List, Dict, Any

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

# ---- Env ----
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "epc-kg-final")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Keyword extraction ----
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "from", "is", "are", "was", "were",
    "what", "which", "who", "how", "why", "does", "do", "about", "reported", "report", "find", "show", "give",
}

def keywords(q: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9µμÅÅ\-]+", q.lower())
    toks = [t for t in toks if t not in STOPWORDS and len(t) >= 3]
    # dedupe while preserving order
    seen = set()
    out: List[str] = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:12]

# ---- Neo4j retrieval ----
RETRIEVAL_CYPHER = """
WITH $terms AS terms, $require_self_trapped AS require_self_trapped
MATCH (e:Evidence)
WITH e,
     reduce(score = 0, t IN terms |
       score + CASE WHEN toLower(e.evidence_quote) CONTAINS t THEN 1 ELSE 0 END
     )
     + CASE WHEN toLower(e.evidence_quote) CONTAINS "self-trapped exciton" THEN 3 ELSE 0 END
     AS score,
     require_self_trapped AS require_self_trapped
WHERE score > 0
  AND (
    NOT require_self_trapped
    OR toLower(e.evidence_quote) CONTAINS "self-trapped"
    OR toLower(e.evidence_quote) CONTAINS "self trapped"
  )
OPTIONAL MATCH (e)-[:HAS_EVIDENCE|:EVIDENCE_FOR]-(ent:Entity)
WHERE ent.type = "Material"
WITH
  e,
  score,
  collect(DISTINCT ent.name) AS materials
RETURN
  e.paper_id AS paper,
  e.chunk_id AS chunk,
  e.rel_type AS rel_type,
  e.evidence_quote AS quote,
  materials AS materials,
  score AS score
ORDER BY score DESC, paper, chunk
LIMIT $limit
"""

def retrieve(driver, question: str, limit: int = 12, database: str = "epc-kg"):
    """
    Retrieve evidence rows from Neo4j for a given question.

    Parameters
    ----------
    driver : neo4j.GraphDatabase.driver
        Neo4j driver instance.
    question : str
        User question.
    limit : int
        Max number of evidence rows to return.
    database : str
        Neo4j database name (e.g., "epc-kg" or "epc-kg-final"). Must match SHOW DATABASES.
    """
    # basic tokenization
    terms = [t.strip().lower() for t in question.split() if t.strip()]

    # enable stricter filtering only for self-trapped questions
    q = question.lower()
    require_self_trapped = ("self-trapped" in q) or ("self trapped" in q)

    with driver.session(database=NEO4J_DATABASE) as session:
        rows = session.run(
            RETRIEVAL_CYPHER,
            terms=terms,
            limit=limit,
            require_self_trapped=require_self_trapped,
        ).data()

        return rows

# ---- Formatting + answering ----
def format_context(rows):
    lines = []
    for i, r in enumerate(rows, start=1):
        mats = r.get("materials") or []
        lines.append(
            f"[{i}]\n"
            f"paper: {r['paper']}\n"
            f"chunk: {r['chunk']}\n"
            f"rel_type: {r['rel_type']}\n"
            f"materials: {', '.join(mats) if mats else 'N/A'}\n"
            f"quote: {r['quote']}\n"
        )
    return "\n".join(lines)

def answer(question: str, rows: List[Dict[str, Any]]) -> str:
    context = format_context(rows)
    system = (
    "Use ONLY the Evidence entries. "
    "If asked 'which materials', list the material names from the Evidence 'materials:' field for entries whose quotes directly support the claim. "
    "Be concise; do not repeat the same conclusion. "
    "Each sentence MUST end with citations using bracketed Evidence IDs like [1]. "
)
    user = f"""Question:
{question}

Evidence (cite by the bracketed ID exactly):
{context}
"""
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    text = resp.choices[0].message.content

    # Optional: basic citation sanity check (IDs must be in range)
    valid = set(range(1, len(rows) + 1))
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    if not cited.issubset(valid):
        raise ValueError(f"Model cited out-of-range evidence IDs: {sorted(cited - valid)}")

    return text

def main() -> None:
    import sys
    if len(sys.argv) < 2:
        print('Usage: python ask_rag.py "your question here"')
        raise SystemExit(2)

    question = sys.argv[1]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        rows = retrieve(driver, question, limit=12, database="epc-kg-final")
        if not rows:
            print(
                "No evidence retrieved. Try adding a material name (e.g., MAPbBr3) "
                "or a parameter (e.g., Frohlich coupling)."
            )
            return

        print("\n=== Retrieved Evidence ===\n")
        print(format_context(rows))

        # DEBUG: show which evidence ID contains "230" (if any)
        for i, r in enumerate(rows, start=1):
            if "230" in str(r.get("quote", "")):
                print(f"\nDEBUG: '230' appears in evidence [{i}] -> {r['quote']}")

        print("\n=== Answer (Grounded) ===\n")
        print(answer(question, rows))
    finally:
        driver.close()


if __name__ == "__main__":
    main()