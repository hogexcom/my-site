+++
date = '2025-11-30T13:08:40+09:00'
draft = false
title = 'Hugoを使った静的サイト構築入門'
tags = ['Hugo', '静的サイト', 'Webサイト']
categories = ['技術']
+++

## Hugoとは

Hugoは、Go言語で書かれた高速な静的サイトジェネレーターです。マークダウンファイルから美しいWebサイトを数秒で生成できます。

## Hugoの特徴

### 1. 圧倒的な速度

Hugoは数千ページのサイトでも1秒以内にビルドできるほど高速です。これは他の静的サイトジェネレーターと比較しても圧倒的な速さです。

### 2. シンプルなセットアップ

単一のバイナリファイルとして配布されているため、依存関係の管理が不要です。インストールも非常に簡単です。

```bash
# macOSの場合
brew install hugo

# バージョン確認
hugo version
```

### 3. 豊富なテーマ

300以上のテーマが用意されており、好みのデザインをすぐに適用できます。今回使用しているPaperModもその一つです。

## 基本的な使い方

### 新しいサイトの作成

```bash
hugo new site my-site
cd my-site
```

### テーマのインストール

```bash
git init
git submodule add https://github.com/adityatelange/hugo-PaperMod themes/PaperMod
```

### 記事の作成

```bash
hugo new posts/my-first-post.md
```

### ローカルサーバーの起動

```bash
hugo server -D
```

ブラウザで `http://localhost:1313` にアクセスすると、サイトがライブプレビューされます。

## まとめ

Hugoは以下のようなサイトに最適です：

- 個人ブログ
- ポートフォリオサイト
- ドキュメントサイト
- 企業のコーポレートサイト

マークダウンで記事を書くだけで、美しく高速なWebサイトが構築できます。ぜひ試してみてください！