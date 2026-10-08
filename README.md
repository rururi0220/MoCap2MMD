# MoCap2MMD

**FBX / BVH to MMD (VMD) Motion Retargeting Desktop Application**  
*オープンソース (MIT License)*

---

## 概要

**MoCap2MMD** は、Mixamo・VRoid等の各種3Dキャラクターアニメーション（FBX）や、mocopi・Perception Neuron・Rokoko等のモーションキャプチャデータ（BVH）を解析し、MMD（MikuMikuDance）モデル（PMX）の骨格構造に合わせて自動リターゲティングを行い、**VMDモーションファイル**を出力するデスクトップアプリケーションです。

軽量GUIフレームワーク（`pywebview`）とWeb標準技術（Three.js）を採用しており、変換結果をリアルタイム3Dプレビューで確認しながら直感的に操作できます。

---

## 主な機能

* **高精度な自動リターゲティング**:
  * **初期ポーズ自動補正**: Aポーズ ↔ Tポーズの差異を自動検出して補正
  * **体格比率スケーリング**: ソースとモデルの脚長比率を計測し、ルート（センター）の移動量を自動スケーリング
  * **足IK / つま先IK 自動逆算**: 足首・足先のワールド軌跡からMMDの足IKキーフレームを自動生成
  * **背骨・腰の多段配分**: 単一Spineの回転を「下半身」「上半身」「上半身2」へ適切に配分
  * **捩りボーン（Twist）分解**: Swing-Twist分解による「腕捩」「手捩」への自動分離
  * **座標系変換**: 右手系（FBX/BVH）から左手系（MMD）への完全な幾何学変換
* **リアルタイム3Dプレビュー**:
  * Three.js (`MMDLoader`) によるPMXモデル描画とモーション即時プレビュー再生
  * 再生・停止・シークバー・ループ再生
* **CLI（コマンドライン）対応**:
  * バッチ処理や自動化パイプラインのためのCLIコマンド（`cli.py`）を標準搭載

---

## システム要件

* **OS**: Windows 10/11 (64bit), macOS, Linux
* **Python**: 3.10 以上 (Python 3.13 動作確認済)

---

## インストール手順

```bash
git clone https://github.com/rururi0220/MoCap2MMD.git
cd MoCap2MMD

# 依存関係のインストール
pip install -r requirements.txt
```

---

## 使い方

### 1. GUI アプリケーションの起動

```bash
python app.py
```

1. **ソースモーション**: 「FBX / BVH ファイルを選択」をクリックしてアニメーションファイルを指定
2. **ターゲットモデル**: 「PMX モデルを選択」をクリックして対象のMMDモデルを指定
3. **変換実行**: 「⚡ 変換実行 & プレビュー」をクリックするとリターゲティングが実行され、中央の3Dビューポートで即時プレビュー再生されます
4. **VMD保存**: 「💾 VMD保存 (.vmd)」をクリックしてVMDファイルをローカルに書き出します

### 2. コマンドライン（CLI）での一括変換

```bash
python cli.py -s input_motion.bvh -t model.pmx -o output.vmd
```

#### オプション一覧

* `-s, --source`: ソースモーションファイルパス (`.fbx` または `.bvh`)
* `-t, --target`: 対象PMXモデルファイルパス (`.pmx`)
* `-o, --output`: 出力先VMDファイルパス (`.vmd`)
* `--scale`: ルート移動スケールの手動倍率（デフォルト: `1.0`）
* `--no-ik`: 足IK生成を無効化し、FK回転で出力
* `--no-twist`: 捩りボーン（腕捩・手捩）の分離を無効化
* `--no-fingers`: 指ボーンの転送を無効化
* `--fps`: 出力フレームレート（デフォルト: `30.0`）

---

## プロジェクト構造

```text
MoCap2MMD/
├── app.py                    # デスクトップGUI起動エントリーポイント (pywebview)
├── cli.py                    # コマンドライン実行エントリーポイント
├── requirements.txt          # Python依存ライブラリ一覧
├── backend/                  # リターゲティング演算コア (Python)
│   ├── api.py                # pywebviewとフロントエンドを繋ぐブリッジクラス
│   ├── skeleton.py           # 共通スケルトン・アニメーションモデル
│   ├── parser/
│   │   ├── bvh_parser.py     # BVH解析
│   │   ├── fbx_parser.py     # ufbx によるFBX解析
│   │   └── pmx_parser.py     # PMXバイナリ読み込み
│   ├── retarget/
│   │   ├── bone_mapping.py   # プリセット辞書照合 & 幾何学ボーン推定
│   │   ├── math_utils.py     # クォータニオン演算, Swing-Twist分解, 座標変換
│   │   ├── ik_solver.py      # 足IK/つま先IKのターゲット座標逆算
│   │   └── retargeter.py     # リターゲティング統合コア
│   └── exporter/
│       └── vmd_exporter.py   # Shift-JIS準拠 VMDバイナリシリアライザ
└── frontend/                 # Webフロントエンド資産 (HTML/CSS/Three.js)
    ├── index.html            # メイン画面マークアップ
    ├── css/
    │   └── style.css         # モダンダークテーマUI
    └── js/
        ├── app.js            # UI操作・バックエンドAPI連携
        ├── viewer.js         # Three.js 3Dプレビュービューポート制御
        └── libs/             # スタンドアロン用Three.js & MMDLoader資産
```

---

## ライセンス

本プロジェクトは [MIT License](LICENSE) のもとで公開されています。
外部依存関係もすべて寛容なオープンソースライセンス（MIT / BSD / Unlicense）で構成されています。