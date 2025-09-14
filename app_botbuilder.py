# app_botbuilder.py
import os
from dotenv import load_dotenv
from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    TurnContext,
    ConversationState,
    MemoryStorage,
    ActivityHandler,
    BotFrameworkAdapter,
)
from botbuilder.schema import Activity
from agents.orchestrator import run_multi_agent_pipeline

load_dotenv()

APP_ID = os.environ.get("BOT_APP_ID", "")
APP_PASSWORD = os.environ.get("BOT_APP_PASSWORD", "")

class TeamsRAGBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        user_text = (turn_context.activity.text or "").strip()
        try:
            answer, citations = await run_multi_agent_pipeline(user_text)
            text = answer
            if citations:
                text += "\n\n---\n参考:\n" + "\n".join(f"- {c['title']} ({c['url']})" for c in citations[:5])
            await turn_context.send_activity(text)
        except Exception as e:
            await turn_context.send_activity(f"エラーが発生しました: {e}")

    async def on_members_added_activity(self, members_added, turn_context: TurnContext):
        await turn_context.send_activity("ようこそ！質問を送ると、RAG＋マルチエージェントで回答します。/help で使い方。")

async def messages(req: web.Request):
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")
    response = await adapter.process_activity(activity, auth_header, bot.on_turn)
    return web.Response(status=201)

adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

memory = MemoryStorage()
conv_state = ConversationState(memory)

bot = TeamsRAGBot()

app = web.Application()
app.router.add_post("/api/messages", messages)
app.router.add_get("/api/messages", lambda _: web.Response(text="OK", status=200))

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("PORT", 3978)))
