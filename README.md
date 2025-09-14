# Teams RAG AutoGen Bot (Python)

このプロジェクトは、Microsoft Teams 上で動作する AI チャットボットの最小構成サンプルです。

## セットアップ

1. 依存関係

```bash
pip install -r requirements-fallback.txt
```

2. .env を設定

```bash
cp .env.example .env
```

3. 起動

```bash
python app_botbuilder.py
```

4. ngrok 公開

```bash
ngrok http 3978
```

Messaging endpoint を `https://76efbf2c9311.ngrok-free.app/api/messages` に設定。

## 構成

- app_botbuilder.py : BotBuilder ベースのフォールバックサーバ
- agents/ : RAG/AutoGen ロジック
- requirements-fallback.txt : フォールバック用依存

**注:** フォールバック構成では `teams-ai` は未使用のため依存から外しました。必要になった場合は `pip install 'teams-ai>=1.4,<2'` を追加してください。
