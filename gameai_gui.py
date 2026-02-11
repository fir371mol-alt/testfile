import os
import tkinter as tk
from tkinter import filedialog, messagebox

import gameai as core


class GameAIGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Texas Showdown - Human vs AI")
        self.engine = None
        self.human_id = 0
        self.num_players = 5
        self.learner = None

        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(fill="x", padx=8, pady=8)

        tk.Label(self.controls_frame, text="Players:").pack(side="left")
        self.players_var = tk.IntVar(value=5)
        tk.OptionMenu(self.controls_frame, self.players_var, 4, 5, 6).pack(side="left", padx=4)

        tk.Label(self.controls_frame, text="Human Seat:").pack(side="left")
        self.human_var = tk.IntVar(value=0)
        self.human_menu = tk.OptionMenu(self.controls_frame, self.human_var, 0, 1, 2, 3, 4, 5)
        self.human_menu.pack(side="left", padx=4)

        self.use_model_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self.controls_frame,
            text="Use trained model for AI",
            variable=self.use_model_var,
        ).pack(side="left", padx=8)

        self.model_path_var = tk.StringVar(value="q_policy.json")
        tk.Entry(self.controls_frame, textvariable=self.model_path_var, width=30).pack(side="left", padx=4)
        tk.Button(self.controls_frame, text="Browse", command=self.pick_model_file).pack(side="left", padx=4)

        tk.Button(self.controls_frame, text="Start New Match", command=self.start_new_match).pack(side="right")

        self.status_label = tk.Label(root, anchor="w", justify="left")
        self.status_label.pack(fill="x", padx=8)

        self.hand_frame = tk.LabelFrame(root, text="Your Legal Moves")
        self.hand_frame.pack(fill="x", padx=8, pady=6)

        self.leader_frame = tk.LabelFrame(root, text="Leader Choice")
        self.leader_frame.pack(fill="x", padx=8, pady=6)

        log_frame = tk.LabelFrame(root, text="Match Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_text = tk.Text(log_frame, height=24)
        self.log_text.pack(fill="both", expand=True)

        self.ai_policies = []
        self.update_status("Start New Match を押してください。")

    def pick_model_file(self):
        path = filedialog.askopenfilename(
            title="Select trained model JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)

    def update_human_menu(self):
        menu = self.human_menu["menu"]
        menu.delete(0, "end")
        for i in range(self.num_players):
            menu.add_command(label=str(i), command=lambda value=i: self.human_var.set(value))
        if self.human_var.get() >= self.num_players:
            self.human_var.set(0)

    def log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def clear_actions(self):
        for frame in (self.hand_frame, self.leader_frame):
            for w in frame.winfo_children():
                w.destroy()

    def update_status(self, extra: str = ""):
        if not self.engine:
            self.status_label.config(text=extra)
            return
        score_line = " | ".join(f"P{p.id}:{p.score}" for p in self.engine.players)
        turn = self.engine.current_player
        self.status_label.config(
            text=f"Round {self.engine.round_number} | Turn: P{turn} | Scores: {score_line} {extra}"
        )

    def load_learner(self):
        self.learner = None
        if not self.use_model_var.get():
            return
        path = self.model_path_var.get().strip()
        if not path:
            return
        if not os.path.exists(path):
            messagebox.showwarning("Model", f"Model file not found: {path}\nRule-based AIで続行します。")
            return
        try:
            self.learner = core.QLearningPolicy.load(path)
            self.log(f"Loaded model: {path}")
        except Exception as exc:
            messagebox.showwarning("Model", f"Failed to load model: {exc}\nRule-based AIで続行します。")
            self.learner = None

    def build_policies(self):
        self.ai_policies = [core.rule_based_policy for _ in range(self.num_players)]
        if self.learner is not None:
            learned_policy = self.learner.as_policy(epsilon=0.0)
            for i in range(self.num_players):
                if i != self.human_id:
                    self.ai_policies[i] = learned_policy

    def start_new_match(self):
        self.num_players = self.players_var.get()
        self.human_id = self.human_var.get()
        if not (0 <= self.human_id < self.num_players):
            self.human_id = 0
            self.human_var.set(0)

        self.update_human_menu()
        self.clear_actions()
        self.log_text.delete("1.0", "end")

        self.load_learner()
        self.build_policies()

        self.engine = core.GameEngine(self.num_players)
        self.engine.start_new_round()
        self.log(f"=== MATCH START (players={self.num_players}, human=P{self.human_id}) ===")
        self.update_status()
        self.advance_until_human_turn()

    def force_zero_if_needed(self):
        if not self.engine:
            return
        if self.engine.trick is None or len(self.engine.trick.plays) == 0:
            pid = self.engine.current_player
            hand = self.engine.players[pid].hand
            zero_cards = [c for c in hand if c.value == 0]
            if zero_cards:
                card = zero_cards[0]
                self.engine.play_card(pid, card)
                self.log(f"P{pid} forced plays {card}")

    def resolve_trick_if_complete(self):
        if self.engine.trick is not None and len(self.engine.trick.plays) == self.engine.num_players:
            winner, color, need_choice = self.engine.resolve_trick()
            self.log(f"→ Trick winner: P{winner} (color={color})")
            if need_choice:
                if winner == self.human_id:
                    self.show_leader_choice(winner)
                    return False
                next_leader = core.choose_leader_rule(self.engine, winner)
                self.engine.choose_next_leader(winner, next_leader)
                self.log(f"P{winner} chooses next leader: P{next_leader}")
        return True

    def show_leader_choice(self, chooser_id: int):
        for w in self.leader_frame.winfo_children():
            w.destroy()
        tk.Label(self.leader_frame, text=f"あなた(P{chooser_id})が次リーダーを選択").pack(side="left", padx=4)
        for pid in self.engine.leader_choice_candidates:
            tk.Button(
                self.leader_frame,
                text=f"P{pid}",
                command=lambda p=pid: self.choose_next_leader(p),
            ).pack(side="left", padx=2)

    def choose_next_leader(self, pid: int):
        chooser = self.engine.leader_choice_by
        self.engine.choose_next_leader(chooser, pid)
        self.log(f"You chose next leader: P{pid}")
        for w in self.leader_frame.winfo_children():
            w.destroy()
        self.advance_until_human_turn()

    def play_human_card(self, card):
        pid = self.engine.current_player
        if pid != self.human_id:
            return
        self.engine.play_card(pid, card)
        self.log(f"You(P{pid}) play {card}")
        self.clear_hand_buttons()
        self.advance_until_human_turn()

    def clear_hand_buttons(self):
        for w in self.hand_frame.winfo_children():
            w.destroy()

    def render_human_actions(self):
        self.clear_hand_buttons()
        legal = self.engine.legal_moves(self.human_id)
        legal_sorted = sorted(legal, key=lambda c: (c.color, c.value))
        if not legal_sorted:
            tk.Label(self.hand_frame, text="合法手なし").pack(anchor="w")
            return

        tk.Label(self.hand_frame, text=f"あなたの合法手 ({len(legal_sorted)}):").pack(anchor="w")
        row = tk.Frame(self.hand_frame)
        row.pack(fill="x")
        for card in legal_sorted:
            tk.Button(
                row,
                text=str(card),
                command=lambda c=card: self.play_human_card(c),
            ).pack(side="left", padx=2, pady=2)

    def advance_until_human_turn(self):
        if not self.engine:
            return

        while True:
            self.force_zero_if_needed()

            if not self.resolve_trick_if_complete():
                self.update_status("(leader choice pending)")
                return

            if self.engine.is_round_over():
                self.engine.end_round()
                self.log("Round finished. " + ", ".join(f"P{i}:+{s}" for i, s in enumerate(self.engine.round_scores)))
                if self.engine.is_match_over():
                    limit = 60 // self.engine.num_players
                    self.log("=== MATCH OVER ===")
                    for p in self.engine.players:
                        result = "LOSE" if p.score >= limit else "WIN"
                        self.log(f"P{p.id}: score={p.score} -> {result}")
                    self.update_status("(match over)")
                    self.clear_hand_buttons()
                    return
                self.engine.start_new_round()
                self.log(f"=== ROUND {self.engine.round_number} START ===")
                continue

            pid = self.engine.current_player
            if pid == self.human_id:
                self.update_status("(your turn)")
                self.render_human_actions()
                return

            legal = self.engine.legal_moves(pid)
            card = self.ai_policies[pid](self.engine, pid, legal)
            self.engine.play_card(pid, card)
            self.log(f"AI(P{pid}) plays {card}")
            self.update_status()


def main():
    root = tk.Tk()
    app = GameAIGUI(root)
    app.update_human_menu()
    root.mainloop()


if __name__ == "__main__":
    main()
