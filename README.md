# testfile

## gameai の学習実行方法

`gameai.py` は単体で実行可能です。

### 1) ルールベース対戦を実行

```bash
python gameai.py --mode play --players 5
```

### 2) Q学習を実行してモデルを保存

```bash
python gameai.py --mode train --players 5 --episodes 500 --matches 30 --model-path q_policy.json
```

- `--episodes`: 学習エピソード数
- `--matches`: 学習後の評価マッチ数
- `--learner-id`: 学習プレイヤーID（既定値 `0`）

### 3) 保存済みモデルを評価

```bash
python gameai.py --mode eval --players 5 --matches 100 --model-path q_policy.json
```

### 4) 学習済みモデルと対戦（1マッチ）

```bash
python gameai.py --mode vs-model --players 5 --learner-id 0 --model-path q_policy.json
```

- `--learner-id` でどの席に学習済みモデルを置くか指定できます。
- `--epsilon` を指定すると、学習済みモデルに探索行動を少し混ぜられます（例: `--epsilon 0.05`）。

学習済みモデルは JSON で保存され、再読み込みして評価/対戦できます。


## GUIで人間 vs AI 対戦

新規ファイル `gameai_gui.py` でGUI対戦できます。

```bash
python gameai_gui.py
```

- Players: 4〜6人
- Human Seat: 自分の席番号
- Use trained model for AI: ONで保存済みモデル（`q_policy.json` など）をAI側に適用
- Start New Match を押すと、あなたの合法手がボタンで表示されます

先に学習する場合の例:

```bash
python gameai.py --mode train --players 5 --episodes 500 --model-path q_policy.json
```
