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

学習済みモデルは JSON で保存され、再読み込みして評価できます。
