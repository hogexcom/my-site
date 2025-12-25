# AGENTS.md

このリポジトリに対する作業メモ（エージェント向け指示）。

## web_python について

- `web_python/` 配下は、Hugo に載せるスタティックアプリをビルドする環境。
- Hugo の `baseURL` に注意すること（例: `hugo.toml` の `baseURL = "https://hogexcom.github.io/my-site/"`）。
  - GitHub Pages（`github.io`）でホストするため、サイトはルート `/` から始まらない（`/my-site/` 配下になる）。
  - **リンクやアセット参照を作るときに**、`/xxx` のようなルート起点のパスを安易に使わず、`baseURL`（またはそれに相当するベースパス）を考慮すること。
  - `web_python/` プロジェクト内でリンクを貼る場合も同様に、ベースパスを考慮すること。

## 実装方針

- 可能な限り Python ファイルを Pyodide で動かす。
- JavaScript での対処は最小限にする（必要な場合のみ補助的に使う）。

