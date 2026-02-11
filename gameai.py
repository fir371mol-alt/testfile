import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# =========================
# 定数定義
# =========================

COLOR_RANGES = {
    "black": range(0, 11),
    "red": range(11, 21),
    "blue": range(21, 30),
    "brown": range(31, 39),
    "green": range(41, 48),
    "yellow": range(51, 57),
    "purple": range(61, 66),
    "gray": range(71, 75),
}

COLOR_MIN = {c: min(r) for c, r in COLOR_RANGES.items()}
COLOR_MAX = {c: max(r) for c, r in COLOR_RANGES.items()}
ALL_VALUES = [v for r in COLOR_RANGES.values() for v in r]


# =========================
# Card
# =========================

class Card:
    def __init__(self, value: int):
        self.value = value
        self.color = self._get_color()

    def _get_color(self):
        for color, r in COLOR_RANGES.items():
            if self.value in r:
                return color
        raise ValueError("Invalid card")

    def __repr__(self):
        return f"{self.color}:{self.value}"


# =========================
# Player
# =========================

class Player:
    def __init__(self, pid: int):
        self.id = pid
        self.hand = []
        self.score = 0  # 累計点（マッチ全体）

    def __repr__(self):
        return f"Player{self.id}(score={self.score}, hand={len(self.hand)})"


# =========================
# Trick
# =========================

class Trick:
    def __init__(self):
        self.plays = []              # (player_id, Card)
        self.color_count = defaultdict(int)
        self.color_max = {}

    def play(self, pid: int, card: Card):
        self.plays.append((pid, card))
        self.color_count[card.color] += 1
        self.color_max[card.color] = max(
            self.color_max.get(card.color, -1),
            card.value
        )


# =========================
# GameEngine
# =========================

class GameEngine:
    def __init__(self, num_players: int):
        assert 4 <= num_players <= 6
        self.num_players = num_players
        self.players = [Player(i) for i in range(num_players)]

        self.current_player = None
        self.trick = None

        self.round_number = 0
        self.round_scores = [0] * num_players

        self.played_cards = defaultdict(set)  # color -> set(values)

        self.leader_choice_pending = False
        self.leader_choice_by = None
        self.leader_choice_candidates = None


    # ---------- マッチ終了判定 ----------
    def is_match_over(self):
        limit = 60 // self.num_players
        return any(p.score >= limit for p in self.players)

    # ---------- ラウンド開始 ----------
    def start_new_round(self):
        self.round_number += 1
        self.round_scores = [0] * self.num_players

        # 山札再生成
        deck = [Card(v) for v in ALL_VALUES]
        random.shuffle(deck)

        for p in self.players:
            p.hand.clear()

        for i, card in enumerate(deck):
            self.players[i % self.num_players].hand.append(card)

        # 0を持っている人が最初
        for p in self.players:
            if any(c.value == 0 for c in p.hand):
                self.current_player = p.id
                break

        self.trick = None
        self.played_cards = defaultdict(set)
        
    # ---------- ラウンド終了判定 ----------
    def is_round_over(self):
        return all(len(p.hand) == 0 for p in self.players)

    # ---------- ラウンド終了 ----------
    def end_round(self):
        for i, p in enumerate(self.players):
            p.score += self.round_scores[i]

    # ---------- 合法手 ----------
    def legal_moves(self, player_id: int):
        player = self.players[player_id]

        if not player.hand:
            return []

        # リード
        if self.trick is None or len(self.trick.plays) == 0:
            return list(player.hand)

        colors_on_table = {c.color for _, c in self.trick.plays}
        same_color_cards = [
            c for c in player.hand if c.color in colors_on_table
        ]

        # 持っていればそれを出す
        if same_color_cards:
            return same_color_cards

        # 持っていなければ何でもOK
        return list(player.hand)

    # ---------- カードを出す ----------
    def play_card(self, player_id: int, card: Card):
        assert player_id == self.current_player
        assert card in self.legal_moves(player_id)

        if self.trick is None:
            self.trick = Trick()

        self.trick.play(player_id, card)
        self.players[player_id].hand.remove(card)

        self.played_cards[card.color].add(card.value)
        self.current_player = (self.current_player + 1) % self.num_players

    # ---------- トリック解決 ----------
    def resolve_trick(self):
        trick = self.trick

        max_count = max(trick.color_count.values())
        candidate_colors = [c for c, cnt in trick.color_count.items() if cnt == max_count]
        winning_color = max(candidate_colors, key=lambda c: trick.color_max[c])

        winner = None
        for pid, card in trick.plays:
            if card.color == winning_color and card.value == trick.color_max[winning_color]:
                winner = pid
                break

        self.round_scores[winner] += 1

        # ★最大値で勝ったか？
        won_with_color_max = (trick.color_max[winning_color] == COLOR_MAX[winning_color])

        self.trick = None

        if won_with_color_max:
            self.leader_choice_pending = True
            self.leader_choice_by = winner
            self.leader_choice_candidates = list(range(self.num_players))  # 仕様に応じて変更可
            # ここでは current_player を確定しない
        else:
            self.current_player = winner

        return winner, winning_color, won_with_color_max

    def choose_next_leader(self, chooser_id: int, next_leader_id: int):
        assert self.leader_choice_pending
        assert chooser_id == self.leader_choice_by
        assert next_leader_id in self.leader_choice_candidates

        self.current_player = next_leader_id
        self.leader_choice_pending = False
        self.leader_choice_by = None
        self.leader_choice_candidates = None
        
    # ---------- 1トリック実行 ----------
    def play_trick(self, policy_fn):
        # 新しいトリック開始
        if self.trick is None:
            self.trick = Trick()

            # 初手が0なら強制
            player = self.players[self.current_player]
            zero_cards = [c for c in player.hand if c.value == 0]
            if zero_cards:
                self.play_card(self.current_player, zero_cards[0])

        while len(self.trick.plays) < self.num_players:
            pid = self.current_player
            legal = self.legal_moves(pid)

            if not legal:
                raise RuntimeError(
                    f"No legal moves for Player {pid}, "
                    f"hand={self.players[pid].hand}, "
                    f"table={[c for _, c in self.trick.plays]}"
                )
            card = policy_fn[pid](self, pid, legal)
            self.play_card(pid, card)

        return self.resolve_trick()

def print_hands(engine):
    print("Hands:")
    for p in engine.players:
        cards = ", ".join(str(c) for c in sorted(p.hand, key=lambda x: (x.color, x.value)))
        print(f"  Player {p.id} ({len(p.hand)} cards): {cards}")

def print_last_turn_outcomes(engine, pid):
    legal = engine.legal_moves(pid)
    for c in legal:
        w = winner_if_last_play(engine, pid, c)
        print(f"Play {c} -> winner Player {w}")

#ログ
def play_trick_verbose(engine, policy_fn, trick_no):
    print(f"\n--- Round {engine.round_number} / Trick {trick_no} ---")
    print_hands(engine)

    engine.trick = Trick()

    # 初手0の強制
    player = engine.players[engine.current_player]
    zero_cards = [c for c in player.hand if c.value == 0]
    if zero_cards:
        card = zero_cards[0]
        print(f"Player {engine.current_player} (forced) plays {card}")
        engine.play_card(engine.current_player, card)

    while len(engine.trick.plays) < engine.num_players:
        pid = engine.current_player
        legal = engine.legal_moves(pid)
        card = policy_fn[pid](engine, pid, legal)

        print(f"Player {pid} plays {card}")
        if engine.trick and len(engine.trick.plays) == engine.num_players - 1:
            if is_forced_take_last(engine, pid):
                print(f"[ForcedTakeLast] Player {pid} will take no matter what.")
                print_last_turn_outcomes(engine, pid)

        engine.play_card(pid, card)

    winner, color, need_choice = engine.play_trick(policy_fn)

    if need_choice:
        chooser = winner  # ここが勝者であること
        next_leader = choose_leader_rule(engine, chooser)
        engine.choose_next_leader(chooser, next_leader)
    print(f"→ Winner: Player {winner} (color={color})")

    return winner

# =========================
# Reinforcement Learning Helper
# =========================

PolicyFn = Callable[["GameEngine", int, List[Card]], Card]


class QLearningPolicy:
    """
    簡易Q学習ポリシー。
    - action: カードvalue(0-74)
    - state: ラフな特徴量タプル（軽量）
    """

    def __init__(self):
        self.q: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    def state_key(self, engine: "GameEngine", player_id: int) -> Tuple:
        player = engine.players[player_id]
        hand_size = len(player.hand)

        # 手札の色枚数（順序固定）
        color_counts = tuple(
            sum(1 for c in player.hand if c.color == color)
            for color in COLOR_RANGES.keys()
        )

        # 場情報（なければNone）
        if engine.trick is None or len(engine.trick.plays) == 0:
            trick_size = 0
            top_color = "none"
            top_value = -1
        else:
            trick_size = len(engine.trick.plays)
            last_card = engine.trick.plays[-1][1]
            top_color = last_card.color
            top_value = last_card.value

        score_gap = engine.players[player_id].score - min(p.score for p in engine.players)

        return (
            hand_size,
            color_counts,
            trick_size,
            top_color,
            top_value,
            score_gap,
        )

    def act(self, engine: "GameEngine", player_id: int, legal_moves: List[Card], epsilon: float = 0.0) -> Card:
        if not legal_moves:
            raise RuntimeError("No legal moves")

        if random.random() < epsilon:
            return random.choice(legal_moves)

        state = self.state_key(engine, player_id)
        # 未知状態はランダムで探索寄りに開始
        if state not in self.q:
            return random.choice(legal_moves)

        return max(legal_moves, key=lambda c: self.q[state][c.value])

    def update(
        self,
        state: Tuple,
        action_value: int,
        reward: float,
        next_state: Tuple,
        next_legal_values: List[int],
        alpha: float,
        gamma: float,
        done: bool,
    ):
        current = self.q[state][action_value]
        if done or not next_legal_values:
            target = reward
        else:
            next_best = max(self.q[next_state][a] for a in next_legal_values)
            target = reward + gamma * next_best
        self.q[state][action_value] = current + alpha * (target - current)

    def to_dict(self):
        rows = []
        for state, actions in self.q.items():
            hand_size, color_counts, trick_size, top_color, top_value, score_gap = state
            rows.append(
                {
                    "state": {
                        "hand_size": hand_size,
                        "color_counts": list(color_counts),
                        "trick_size": trick_size,
                        "top_color": top_color,
                        "top_value": top_value,
                        "score_gap": score_gap,
                    },
                    "actions": {str(a): q for a, q in actions.items()},
                }
            )
        return {"q_table": rows}

    @classmethod
    def from_dict(cls, data):
        model = cls()
        for row in data.get("q_table", []):
            st = row["state"]
            state = (
                int(st["hand_size"]),
                tuple(int(x) for x in st["color_counts"]),
                int(st["trick_size"]),
                st["top_color"],
                int(st["top_value"]),
                int(st["score_gap"]),
            )
            for a, qv in row["actions"].items():
                model.q[state][int(a)] = float(qv)
        return model

    def save(self, path: str):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False))

    @classmethod
    def load(cls, path: str):
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def as_policy(self, epsilon: float = 0.0) -> PolicyFn:
        def _policy(engine: "GameEngine", player_id: int, legal_moves: List[Card]) -> Card:
            return self.act(engine, player_id, legal_moves, epsilon=epsilon)

        return _policy


def train_q_policy(
    episodes: int = 500,
    num_players: int = 5,
    learner_id: int = 0,
    alpha: float = 0.1,
    gamma: float = 0.98,
    epsilon_start: float = 0.30,
    epsilon_end: float = 0.05,
):
    """
    学習対象(learner_id)だけQ学習し、他プレイヤーはルールベースで固定。
    報酬設計:
      - トリック獲得 +1
      - ラウンド終了時の獲得点数を加算
      - マッチ勝利 +5 / 敗北 -5
    """
    learner = QLearningPolicy()

    for ep in range(1, episodes + 1):
        engine = GameEngine(num_players)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / episodes)

        while not engine.is_match_over():
            engine.start_new_round()

            while not engine.is_round_over():
                state = None
                action_value = None

                if engine.current_player == learner_id:
                    legal = engine.legal_moves(learner_id)
                    state = learner.state_key(engine, learner_id)
                    card = learner.act(engine, learner_id, legal, epsilon=epsilon)
                    action_value = card.value
                    engine.play_card(learner_id, card)
                else:
                    pid = engine.current_player
                    legal = engine.legal_moves(pid)
                    card = rule_based_policy(engine, pid, legal)
                    engine.play_card(pid, card)

                # トリック終了タイミングで報酬更新
                if engine.trick is not None and len(engine.trick.plays) == engine.num_players:
                    winner, _, need_choice = engine.resolve_trick()
                    if need_choice:
                        next_leader = choose_leader_rule(engine, winner)
                        engine.choose_next_leader(winner, next_leader)

                    if state is not None:
                        reward = 1.0 if winner == learner_id else 0.0
                        done = engine.is_round_over() and engine.is_match_over()

                        if not done:
                            next_state = learner.state_key(engine, learner_id)
                            next_legal = [c.value for c in engine.legal_moves(learner_id)]
                        else:
                            next_state = state
                            next_legal = []

                        learner.update(
                            state=state,
                            action_value=action_value,
                            reward=reward,
                            next_state=next_state,
                            next_legal_values=next_legal,
                            alpha=alpha,
                            gamma=gamma,
                            done=done,
                        )

            # ラウンド終了
            round_reward = engine.round_scores[learner_id]
            engine.end_round()

            # 直近状態へラウンド報酬を反映（簡易）
            if learner.q:
                # 最新更新状態に薄く加点
                any_state = next(reversed(learner.q))
                for a in learner.q[any_state]:
                    learner.q[any_state][a] += 0.01 * round_reward

        # マッチ終了ボーナス
        limit = 60 // engine.num_players
        match_win = engine.players[learner_id].score < limit
        bonus = 5.0 if match_win else -5.0
        if learner.q:
            any_state = next(reversed(learner.q))
            for a in learner.q[any_state]:
                learner.q[any_state][a] += 0.01 * bonus

        if ep % 50 == 0:
            print(f"[train] episode={ep} epsilon={epsilon:.3f} q_states={len(learner.q)}")

    return learner


def build_policies_with_learner(
    num_players: int,
    learner: QLearningPolicy,
    learner_id: int = 0,
    epsilon: float = 0.0,
):
    policies = [rule_based_policy for _ in range(num_players)]
    policies[learner_id] = learner.as_policy(epsilon=epsilon)
    return policies


def evaluate_learner(
    learner: QLearningPolicy,
    matches: int = 30,
    num_players: int = 5,
    learner_id: int = 0,
):
    wins = 0
    total_score = 0
    for _ in range(matches):
        policies = build_policies_with_learner(num_players, learner, learner_id=learner_id, epsilon=0.0)
        result = run_match(num_players=num_players, policies=policies, verbose=False)
        limit = 60 // num_players
        if result.players[learner_id].score < limit:
            wins += 1
        total_score += result.players[learner_id].score

    return {
        "matches": matches,
        "win_rate": wins / matches if matches else 0.0,
        "avg_score": total_score / matches if matches else 0.0,
    }


# =========================
# rulebased_ai
# =========================
def choose_leader_rule(engine, chooser_id):
    limit = 60 // engine.num_players

    
    # 3) それでも決まらないなら時計回り（chooserの次）
    return (chooser_id + 1) % engine.num_players


def random_policy(engine, player_id, legal_moves):
    return random.choice(legal_moves)

def color_values(color):
    return list(COLOR_RANGES[color])

def is_color_exhausted_for_others(engine, player_id, color):
    # その色の全カードが「プレイ済み or 自分の手札」にあるなら、
    # 他人の手札にその色は存在しない＝色が枯れている
    my_vals = {c.value for c in engine.players[player_id].hand if c.color == color}
    played = engine.played_cards[color]
    all_vals = set(color_values(color))
    return all_vals.issubset(played | my_vals)

def winner_if_last_play(engine, pid, card):
    """
    現在のトリックが「あと1手」で、pidがcardを出したと仮定したときの勝者IDを返す。
    ※engine状態は変更しない（コピー計算）
    """
    trick = engine.trick
    assert trick is not None
    assert len(trick.plays) == engine.num_players - 1
    assert engine.current_player == pid

    # 現在の集計をコピー
    color_count = dict(trick.color_count)
    color_max = dict(trick.color_max)

    # pidがcardを出した場合を反映
    color_count[card.color] = color_count.get(card.color, 0) + 1
    color_max[card.color] = max(color_max.get(card.color, -1), card.value)

    # 勝ち色決定：最頻色、同数ならその色の最大値が高い方
    max_count = max(color_count.values())
    candidate_colors = [c for c, cnt in color_count.items() if cnt == max_count]
    winning_color = max(candidate_colors, key=lambda c: color_max[c])

    # その勝ち色の最大値を出したプレイヤーが勝者
    winning_value = color_max[winning_color]

    # trick.plays に最後の一手を足した全プレイ列から、該当カードのプレイヤーを探す
    all_plays = trick.plays + [(pid, card)]
    for p, c in all_plays:
        if c.color == winning_color and c.value == winning_value:
            return p

    raise RuntimeError("winner calculation failed")

def is_forced_take_last(engine, pid):
    """
    最終手番のときに限り、
    合法手のどれを出しても自分がトリックを取るなら True
    """
    if engine.trick is None:
        return False
    if engine.current_player != pid:
        return False
    if len(engine.trick.plays) != engine.num_players - 1:
        return False

    legal = engine.legal_moves(pid)
    if not legal:
        return False

    # どの合法手でも勝者がpidなら強制取得
    return all(winner_if_last_play(engine, pid, c) == pid for c in legal)

def pick_highest_risk_or_color_max(engine, pid):
    """
    最終手番で forced take が確定している状況で使う。
    ルール：
      1) legal_moves の中に「色の最大値」があれば、その中から危険度最大を出す
      2) 無ければ legal_moves の中で危険度最大を出す
    """
    legal = engine.legal_moves(pid)

    # 1) 色の最大値カード（例：gray:74, purple:65 ...）が合法手にあるか
    color_max_cards = [c for c in legal if c.value == COLOR_MAX[c.color]]
    if color_max_cards:
        return max(color_max_cards, key=lambda c: card_risk(engine, pid, c))

    # 2) 無ければ危険度最大
    return max(legal, key=lambda c: card_risk(engine, pid, c))


#単色孤立ハイカード・ペナルティ
def singleton_high_penalty(engine, player_id, card):
    # その色を何枚持っているか
    my_same = [c for c in engine.players[player_id].hand if c.color == card.color]
    if len(my_same) != 1:
        return 0.0

    # 同色内でどれくらい上位か（0..1）
    # 未プレイ集合ベースにするとあなたの危険度(1)と整合が良い
    all_vals = set(COLOR_RANGES[card.color])
    played = engine.played_cards[card.color]
    unplayed = sorted(all_vals - played)

    if not unplayed:
        return 0.0

    idx = {v: i for i, v in enumerate(unplayed)}
    denom = max(1, len(unplayed) - 1)
    highness = idx.get(card.value, len(unplayed) - 1) / denom  # 小=0, 大=1

    # 上位ほど強く罰する（2乗で上位寄りに）
    return 0.8 * (highness ** 2)  # ←係数は調整ポイント

def card_risk(engine, player_id, card):
    color = card.color

    # (2) 色が枯れているなら安全
    if is_color_exhausted_for_others(engine, player_id, color):
        return 0.0

    all_vals = set(color_values(color))
    played = engine.played_cards[color]
    unplayed = sorted(all_vals - played)  # 未プレイ（自分の手札も含む）

    if not unplayed:
        return 0.0

    # 未プレイの中での相対順位（大きいほど危険）
    # 例：最小→0付近、最大→1付近
    rank = unplayed.index(card.value) if card.value in unplayed else len(unplayed) - 1
    size_factor = rank / max(1, (len(unplayed) - 1))

    # 未プレイ枚数が多いほど危険（0〜1）
    remain_factor = len(unplayed) / len(all_vals)

    # 危険度（例：線形合成）
    base = 0.7 * size_factor + 0.3 * remain_factor

    # ★追加：1枚しかない色の高札をより危険に
    base += singleton_high_penalty(engine, player_id, card)

    return base

def trick_info(engine):
    trick = engine.trick
    colors_on_table = set()
    color_count = defaultdict(int)
    color_max = {}

    for _, c in trick.plays:
        colors_on_table.add(c.color)
        color_count[c.color] += 1
        color_max[c.color] = max(color_max.get(c.color, -1), c.value)

    max_count = max(color_count.values()) if color_count else 0
    most_colors = [c for c, cnt in color_count.items() if cnt == max_count]
    most_max_value = max(color_max[c] for c in most_colors) if most_colors else -1

    remaining_players = engine.num_players - len(trick.plays)
    return colors_on_table, color_count, color_max, max_count, most_colors, most_max_value, remaining_players


def is_card_absolutely_safe(engine, player_id, card):
    # トリックが無い（リード）時は、この3条件は使えないので「保証なし」
    # ※必要ならリード専用の絶対安全定義を別途作る
    if engine.trick is None or len(engine.trick.plays) == 0:
        return False

    colors_on_table, color_count, color_max, max_count, most_colors, most_max_value, remaining_players = trick_info(engine)

    # 条件1：すでにその色で自分より大きいカードが場に出ているなら安全
    # （同色が勝ち色になっても、自分が最高値になれない）
    if card.color in color_max and color_max[card.color] > card.value:
        return True

    # 条件3：最頻色の枚数が残りプレイヤー人数より多い
    # → 新しい色（場に無い色）はもう追いつけないので何を出しても自分は取れない
    if max_count > remaining_players:
        if card.color not in colors_on_table:
            return True

    # 条件2：最頻色の枚数が残りプレイヤー人数と同じ
    # → 新しい色で同数タイになる可能性があるが、
    #    「最頻色側の最大値 > 自分のカード」なら、タイでも最頻色が勝つ。
    #    さらに、もし他人が新色で自分より高いのを出して勝っても自分は取らない。
    if max_count == remaining_players:
        if card.color not in colors_on_table and card.value < most_max_value:
            return True

    return False

#上位抱え度
def upper_heaviness(engine, player_id, color):
    # 未プレイ集合（このラウンドでまだ出ていない値）
    all_vals = sorted(COLOR_RANGES[color])
    played = engine.played_cards[color]
    unplayed = [v for v in all_vals if v not in played]
    if not unplayed:
        return 0.0

    # 自分の手札のその色
    my_vals = sorted([c.value for c in engine.players[player_id].hand if c.color == color])
    if not my_vals:
        return 0.0

    # 未プレイの中での“順位の高さ”を [0..1] に正規化
    # 例：最大値に近いほど 1 に近い
    idx = {v: i for i, v in enumerate(unplayed)}
    denom = max(1, len(unplayed) - 1)

    # 上位をどれだけ抱えてるかを見るため「最大」と「上位2枚平均」を混ぜる
    ranks = [(idx.get(v, len(unplayed)-1) / denom) for v in my_vals]  # 小さいほど0, 大きいほど1
    ranks.sort()

    top1 = ranks[-1]
    top2 = (ranks[-1] + ranks[-2]) / 2 if len(ranks) >= 2 else ranks[-1]

    # 係数は好みで調整OK
    return 0.6 * top1 + 0.4 * top2

def is_new_color_play(engine, card):
    if engine.trick is None or len(engine.trick.plays) == 0:
        return True  # リード
    colors_on_table = {c.color for _, c in engine.trick.plays}
    return card.color not in colors_on_table


def rule_based_policy(engine, player_id, legal_moves):
     # 最終手番か？
    if engine.trick and len(engine.trick.plays) == engine.num_players - 1 and engine.current_player == player_id:
        # forced take が確定しているなら、この単純ルール
        if is_forced_take_last(engine, player_id):
            return pick_highest_risk_or_color_max(engine, player_id)

    abs_safe = [c for c in legal_moves if is_card_absolutely_safe(engine, player_id, c)]
    if abs_safe:
        # 絶対安全なら「危険なカードを捨てる」＝危険度が高いものから
        return max(abs_safe, key=lambda c: card_risk(engine, player_id, c))

    def score(card):
        # ベース：カード危険度（小さいほど良い）
        s = card_risk(engine, player_id, card)

        # 新色を出すなら「上位抱え色」を避けるペナルティ
        if is_new_color_play(engine, card):
            s += 1.2 * upper_heaviness(engine, player_id, card.color)  # ←重み調整ポイント

            # 新色で出すなら、その色の中でも低い札を選びやすくする（任意）
            s += 0.2 * (card.value / COLOR_MAX[card.color])

        return s

    # 絶対安全が無いなら、危険度が一番低いもの
    return min(legal_moves, key=lambda c: card_risk(engine, player_id, c))



def run_match(num_players=5, policies=None, verbose=True):
    if policies is None:
        policies = [rule_based_policy for _ in range(num_players)]

    engine = GameEngine(num_players)

    while not engine.is_match_over():
        engine.start_new_round()
        trick_no = 1

        if verbose:
            print(f"\n========== ROUND {engine.round_number} START ==========")

        while not engine.is_round_over():
            if verbose:
                play_trick_verbose(engine, policies, trick_no)
            else:
                winner, _, need_choice = engine.play_trick(policies)
                if need_choice:
                    next_leader = choose_leader_rule(engine, winner)
                    engine.choose_next_leader(winner, next_leader)
            trick_no += 1

        engine.end_round()

        if verbose:
            print("\nRound result:")
            for i, s in enumerate(engine.round_scores):
                print(f"Player {i}: +{s} points")

    if verbose:
        print("=== MATCH OVER ===")
        limit = 60 // engine.num_players
        for p in engine.players:
            result = "LOSE" if p.score >= limit else "WIN"
            print(f"Player {p.id}: score={p.score} → {result}")

    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Texas Showdown AI utility")
    parser.add_argument("--mode", choices=["play", "train", "eval"], default="play")
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--matches", type=int, default=30)
    parser.add_argument("--learner-id", type=int, default=0)
    parser.add_argument("--model-path", default="q_policy.json")
    args = parser.parse_args()

    if args.mode == "play":
        run_match(num_players=args.players, verbose=True)
    elif args.mode == "train":
        learner = train_q_policy(
            episodes=args.episodes,
            num_players=args.players,
            learner_id=args.learner_id,
        )
        learner.save(args.model_path)
        print(f"saved model to: {args.model_path}")
        stats = evaluate_learner(
            learner,
            matches=args.matches,
            num_players=args.players,
            learner_id=args.learner_id,
        )
        print(f"eval stats: {stats}")
    else:
        learner = QLearningPolicy.load(args.model_path)
        stats = evaluate_learner(
            learner,
            matches=args.matches,
            num_players=args.players,
            learner_id=args.learner_id,
        )
        print(f"eval stats: {stats}")
