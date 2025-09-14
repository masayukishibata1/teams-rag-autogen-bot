# agents/orchestrator.py  (AutoGen v0.4 対応版)
import os
from typing import Tuple, List, Dict

# ==== AutoGen v0.4 ====
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import ChatMessage, AgentEvent, TextMessage
from autogen_agentchat.base import TaskResult

# モデルクライアント（OpenAI / Azure OpenAI 両対応）
from autogen_ext.models.openai import (
    OpenAIChatCompletionClient,
    AzureOpenAIChatCompletionClient,
)

from agents.retriever import search_chunks
from dotenv import load_dotenv

load_dotenv() 

def _make_model_client():
    api_type = (os.getenv("OPENAI_API_TYPE", "") or "").lower()
    api_key = os.getenv("OPENAI_API_KEY", "")

    base_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()          # ← 基礎モデル名
    deployment = os.getenv("OPENAI_DEPLOYMENT", "").strip()                # ← デプロイ名

    common = {"seed": 42, "temperature": 0}

    if api_type == "azure":
        endpoint = os.getenv("OPENAI_API_BASE", "").strip()
        api_version = os.getenv("OPENAI_API_VERSION", "2024-08-01-preview").strip()
        if not (endpoint and api_key and deployment and base_model):
            raise RuntimeError(
                "Azure OpenAI の設定不備です。OPENAI_API_BASE / OPENAI_API_KEY / "
                "OPENAI_DEPLOYMENT（デプロイ名）/ OPENAI_MODEL（基礎モデル名）を確認してください。"
            )
        # 重要: model は “基礎モデル名”、azure_deployment は “デプロイ名”
        return AzureOpenAIChatCompletionClient(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
            model=base_model,
            azure_deployment=deployment,
            **common,
        )
    else:
        # 通常の OpenAI: model は基礎モデル名でOK
        return OpenAIChatCompletionClient(
            model=base_model,
            api_key=api_key,
            **common,
        )


def _context_text(docs: List[Dict], max_chars: int = 8000) -> str:
    if not docs:
        return "No context."
    joined = "\n\n".join(f"[{i+1}] {d.get('chunk','')}" for i, d in enumerate(docs))
    if len(joined) > max_chars:
        joined = joined[: max_chars - 200] + "\n\n...[truncated]..."
    return joined


async def run_multi_agent_pipeline(user_query: str) -> Tuple[str, List[Dict]]:
    # ---- RAG: Azure AI Search ----
    docs = search_chunks(user_query, k=5)
    context = _context_text(docs)

    model_client = _make_model_client()

    # ---- Agents (v0.4: AssistantAgent) ----
    retriever = AssistantAgent(
        name="RetrieverAgent",
        description="Collects and summarizes facts from provided CONTEXT.",
        system_message=(
            "You retrieve and summarize relevant snippets for the question. "
            "Return a compact bullet list of facts with doc indices like [1], [2]. "
            "Do not invent content not present in CONTEXT."
        ),
        model_client=model_client,
    )

    answerer = AssistantAgent(
        name="AnswerAgent",
        description="Writes grounded answers using the given CONTEXT.",
        system_message=(
            "You are a helpful enterprise assistant. Answer strictly grounded in the provided CONTEXT. "
            "If the answer isn't in context, say you don't know and suggest where to look. "
            "Cite context indices like [1], [2] for key claims."
        ),
        model_client=model_client,
    )

    critic = AssistantAgent(
        name="CriticAgent",
        description="Reviews and shortens the final answer; enforces citations.",
        system_message=(
            "You check the Answer for hallucinations and missing citations. "
            "Ensure each claim maps to context indices. "
            "Respond with a short, corrected FINAL ANSWER only."
        ),
        model_client=model_client,
    )

    # ---- Team (v0.4: RoundRobinGroupChat) ----
    # 会話は retriever -> answerer -> critic のラウンドロビンで進む / 上限を 6 メッセージに制限
    group = RoundRobinGroupChat(
        [retriever, answerer, critic],
        termination_condition=MaxMessageTermination(max_messages=6),
    )

    # ---- 実行（タスク文字列を渡す）----
    seed = (
        f"USER QUESTION:\n{user_query}\n\n"
        f"CONTEXT (snippets):\n{context}\n\n"
        "Flow:\n"
        "1) RetrieverAgent -> key facts with [index]\n"
        "2) AnswerAgent -> draft answer citing [index]\n"
        "3) CriticAgent -> check & output FINAL ANSWER only."
    )

    # run() は TaskResult を返す。最終発話は messages の末尾。
    result: TaskResult = await group.run(task=seed)

    # 最終テキスト抽出（最後の TextMessage を拾う）
    final_text = ""
    for m in reversed(result.messages):
        if isinstance(m, TextMessage):
            final_text = m.content or ""
            break
        if isinstance(m, ChatMessage) and isinstance(m.content, str):
            final_text = m.content
            break
        if isinstance(m, AgentEvent) and isinstance(m.content, str):
            final_text = m.content
            break
    if not final_text:
        final_text = "（回答を生成できませんでした。クエリを言い換えて再試行してください。）"

    citations = [{"title": d.get("title", "doc"), "url": d.get("url", "")} for d in docs]
    return final_text, citations
