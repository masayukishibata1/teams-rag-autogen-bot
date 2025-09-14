# agents/retriever.py
import os
from typing import List, Dict, Any
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.core.exceptions import HttpResponseError  # 追加
from dotenv import load_dotenv

load_dotenv()  # .env をロード
USE_MANAGED_ID = os.getenv("AZURE_SEARCH_API_KEY") in (None, "", "managed")
SEMANTIC_CONFIG = os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "").strip()

def _client() -> SearchClient:
    endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    index = os.environ["AZURE_SEARCH_INDEX"]
    if USE_MANAGED_ID:
        cred = DefaultAzureCredential()
    else:
        from azure.core.credentials import AzureKeyCredential
        cred = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])
    return SearchClient(endpoint=endpoint, index_name=index, credential=cred)

def search_chunks(query: str, k: int = 5) -> List[Dict[str, Any]]:
    client = _client()
    results = None

    # 1) Semantic 検索（設定があるときだけ）
    if SEMANTIC_CONFIG:
        try:
            results = client.search(
                query,
                top=k,
                query_type="semantic",
                semantic_configuration_name=SEMANTIC_CONFIG,
                include_total_count=False,
            )
        except HttpResponseError:
            # 2) フォールバック：通常検索
            results = client.search(query, top=k)
    else:
        # 3) 最初から通常検索
        results = client.search(query, top=k)

    docs: List[Dict[str, Any]] = []
    for r in results:
        docs.append({
            "score": r.get("@search.score", 0.0),
            "title": r.get("title") or r.get("parent_id") or "doc",
            "chunk": r.get("chunk") or r.get("content") or (r.get("text") or ""),
            "url": r.get("url") or "",
        })
    return docs
