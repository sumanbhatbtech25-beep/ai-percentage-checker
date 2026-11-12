"""Find the research papers most similar to a document.

1. Extract key terms locally (only these short keyword queries leave the
   machine; the document itself is never sent anywhere).
2. Pull candidate papers from OpenAlex and Semantic Scholar (both free).
3. Re-rank all candidates locally by semantic similarity between the full
   document and each paper's title + abstract (BAAI/bge-small-en-v1.5).
"""
from __future__ import annotations

import json
import os
import re
import threading
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .lexicon import AI_PHRASES, STOPWORDS
from .models import DEVICE

EMBED_MODEL = os.environ.get("AIC_EMBED", "BAAI/bge-small-en-v1.5")
OPENALEX_KEY = os.environ.get("OPENALEX_API_KEY", "")
S2_KEY = os.environ.get("S2_API_KEY", "")
UA = {"User-Agent": "ai-percentage-checker/1.0"}

GENERIC = set(
    """study studies paper research results result approach approaches method methods analysis data
    based using used use uses provide provides provided various different important however within
    also however therefore thus first second third new one two many several among across including
    include includes well work works way ways can could would should may might must need needs
    number part area areas time years year make makes made role level levels high low large small
    significant significantly possible potential current recent overall order find findings found
    show shows shown present presents presented paper's author authors article chapter section
    example examples case cases important key main major general specific particular often""".split()
)

_embed = None
_embed_lock = threading.Lock()


def _get_embedder():
    global _embed
    with _embed_lock:
        if _embed is None:
            tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
            model = AutoModel.from_pretrained(EMBED_MODEL).to(DEVICE).eval()
            _embed = (tok, model)
    return _embed


@torch.inference_mode()
def embed(texts: list[str], batch_size: int = 32) -> torch.Tensor:
    tok, model = _get_embedder()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i : i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        cls = model(**enc).last_hidden_state[:, 0]  # bge uses CLS pooling
        out.append(F.normalize(cls.float(), dim=-1).cpu())
    return torch.cat(out) if out else torch.empty(0)


def embed_document(text: str) -> torch.Tensor:
    words = text.split()
    chunks = [" ".join(words[i : i + 350]) for i in range(0, max(len(words), 1), 350)][:12]
    return F.normalize(embed(chunks).mean(0), dim=-1)


# --------------------------------------------------------------------------- keywords
def keywords(text: str, k: int = 10) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text)
    banned = STOPWORDS | GENERIC | set(AI_PHRASES)

    def ok(w: str) -> bool:
        lw = w.lower()
        return lw not in banned and (len(lw) >= 4 or (w.isupper() and 2 <= len(w) <= 6))

    uni = Counter(w if w.isupper() else w.lower() for w in tokens if ok(w))
    for acro in re.findall(r"\b[A-Z][A-Z0-9]{1,5}s?\b", text):  # AI, LLM, IoT-style acronyms
        acro = acro.rstrip("s") if len(acro) > 2 else acro
        if acro.lower() not in STOPWORDS:
            uni[acro] += 2
    bi = Counter()
    for a, b in zip(tokens, tokens[1:]):
        if ok(a) and ok(b):
            bi[f"{a.lower()} {b.lower()}"] += 1
    scored = [(c * 2.5, p) for p, c in bi.items() if c >= 2]
    scored += [(c * (1 + 0.08 * len(w)), w) for w, c in uni.items()]
    scored.sort(reverse=True)
    out: list[str] = []
    for _, term in scored:
        if any(term in t or t in term for t in out):
            continue
        out.append(term)
        if len(out) >= k:
            break
    return out


def build_queries(text: str, kws: list[str]) -> list[str]:
    queries = []
    first = text.strip().split("\n", 1)[0].strip()
    if 3 <= len(first.split()) <= 18 and not first.endswith("."):
        queries.append(first)  # looks like a title
    if kws:
        queries.append(" ".join(kws[:4]))
        queries.append(" ".join(kws[:7]))
    return list(dict.fromkeys(q for q in queries if q))[:3]


# --------------------------------------------------------------------------- sources
def _get(url: str, headers: dict | None = None, timeout: int = 20):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _abstract_from_index(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return " ".join(w for _, w in pos)


def search_openalex(query: str, n: int = 25) -> list[dict]:
    params = {
        "search": query,
        "filter": "has_abstract:true,type:article|preprint|review|book-chapter|dissertation",
        "per_page": n,
        "select": "id,title,publication_year,doi,cited_by_count,authorships,primary_location,open_access,abstract_inverted_index",
    }
    if OPENALEX_KEY:
        params["api_key"] = OPENALEX_KEY
    data = _get("https://api.openalex.org/works?" + urllib.parse.urlencode(params))
    out = []
    for w in data.get("results", []):
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        out.append({
            "title": w.get("title") or "",
            "abstract": _abstract_from_index(w.get("abstract_inverted_index")),
            "year": w.get("publication_year"),
            "authors": [a["author"]["display_name"] for a in (w.get("authorships") or [])[:6] if a.get("author")],
            "venue": src.get("display_name") or "",
            "citations": w.get("cited_by_count") or 0,
            "url": w.get("doi") or loc.get("landing_page_url") or w.get("id"),
            "pdf": (w.get("open_access") or {}).get("oa_url") or "",
            "source": "OpenAlex",
        })
    return out


def search_semantic_scholar(query: str, n: int = 25) -> list[dict]:
    params = {
        "query": query, "limit": n,
        "fields": "title,abstract,year,authors,venue,citationCount,externalIds,url,openAccessPdf",
    }
    headers = {"x-api-key": S2_KEY} if S2_KEY else {}
    data = _get("https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params), headers)
    out = []
    for p in data.get("data", []) or []:
        if not p.get("abstract"):
            continue
        doi = (p.get("externalIds") or {}).get("DOI")
        out.append({
            "title": p.get("title") or "",
            "abstract": p["abstract"],
            "year": p.get("year"),
            "authors": [a.get("name", "") for a in (p.get("authors") or [])[:6]],
            "venue": p.get("venue") or "",
            "citations": p.get("citationCount") or 0,
            "url": f"https://doi.org/{doi}" if doi else p.get("url"),
            "pdf": (p.get("openAccessPdf") or {}).get("url") or "",
            "source": "Semantic Scholar",
        })
    return out


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())[:80]


# --------------------------------------------------------------------------- main
def find_similar(text: str, top: int = 5, query: str | None = None) -> dict:
    text = text.strip()
    from .compare import _sentences  # lazy: compare imports this module

    prose = " ".join(_sentences(text, limit=400))  # no cover page, headings, references or declarations
    kws = keywords(prose if len(prose.split()) >= 40 else text)
    queries = [query.strip()] if query and query.strip() else build_queries(text, kws)
    if not queries:
        return {"keywords": kws, "queries": [], "papers": [], "errors": ["Not enough text to search."]}

    jobs = [(search_openalex, q) for q in queries] + [(search_semantic_scholar, queries[0])]
    errors, pool_results = [], []
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [(fn.__name__, q, ex.submit(fn, q)) for fn, q in jobs]
        for name, q, f in futs:
            try:
                pool_results.extend(f.result())
            except Exception as e:  # rate limits, network
                src = "OpenAlex" if "openalex" in name else "Semantic Scholar"
                errors.append(f"{src} unavailable for “{q}”: {getattr(e, 'code', '') or e}")

    seen, cands = set(), []
    for p in pool_results:
        key = _norm_title(p["title"])
        if not key or key in seen or len(p["abstract"].split()) < 30:
            continue
        seen.add(key)
        cands.append(p)
    if not cands:
        return {"keywords": kws, "queries": queries, "papers": [], "errors": errors or ["No papers found."]}

    doc_vec = embed_document(text)
    vecs = embed([f"{p['title']}. {p['abstract']}" for p in cands])
    sims = (vecs @ doc_vec).tolist()
    kw_low = [k.lower() for k in kws]
    for p, s in zip(cands, sims):
        p["similarity"] = round(s, 4)
        blob = (p["title"] + " " + p["abstract"]).lower()
        p["shared_terms"] = [k for k in kw_low if k in blob][:6]
        p["abstract_full"] = p["abstract"]  # kept whole for the comparison step
        if len(p["abstract"]) > 900:
            p["abstract"] = p["abstract"][:900].rsplit(" ", 1)[0] + "…"
    cands.sort(key=lambda p: p["similarity"], reverse=True)
    return {
        "keywords": kws,
        "queries": queries,
        "candidates_considered": len(cands),
        "papers": cands[:top],
        "errors": errors,
    }
