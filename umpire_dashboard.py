# umpire_dashboard.py  ─  ShuttleEye v5  ─  Umpire Scoring Dashboard
# ═══════════════════════════════════════════════════════════════════════
#
#  A professional Tkinter umpire dashboard that:
#  • Shows live score for both players / teams
#  • Tracks sets (games) with full set history
#  • Receives IN/OUT decisions from line_judge via callback
#  • Shows a live rally log with colour-coded IN/OUT entries
#  • Challenge system: each side gets 2 challenges per set
#    (challenges lost on wrong, retained on correct)
#  • Umpire override buttons: manually correct any call
#  • Player name editing
#  • Serve indicator (which side is serving)
#  • Match timer
#  • Export rally log to text file
#  • Runs in its own thread — non-blocking to the CV loop
#
#  USAGE
#  ─────
#  from umpire_dashboard import UmpireDashboard
#  dash = UmpireDashboard()
#  dash.start()                          # opens window in background thread
#
#  # From line_judge callback or app loop:
#  dash.push_decision("IN",  (rx,ry), (px,py))
#  dash.push_decision("OUT", (rx,ry), (px,py))
#
#  # Score-only update (no line decision):
#  dash.add_point("A")   # or "B"
#
#  dash.stop()                           # close window
# ═══════════════════════════════════════════════════════════════════════

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
import datetime
import queue


# ── Colour palette ────────────────────────────────────────────────────
BG          = "#0d1117"
BG2         = "#161b22"
BG3         = "#21262d"
BORDER      = "#30363d"
TEXT        = "#e6edf3"
TEXT_DIM    = "#8b949e"
GREEN       = "#3fb950"
RED         = "#f85149"
YELLOW      = "#d29922"
CYAN        = "#58a6ff"
ORANGE      = "#f0883e"
WHITE       = "#ffffff"
SCORE_A_COL = "#58a6ff"
SCORE_B_COL = "#f0883e"


class UmpireDashboard:
    """
    Umpire scoring dashboard.
    Runs Tkinter in a dedicated daemon thread so it never blocks the CV loop.
    """

    WINNING_SCORE  = 21
    DEUCE_SCORE    = 20
    MAX_SCORE      = 30
    BEST_OF        = 3          # match is best of 3 sets
    CHALLENGES_PER_SET = 2

    def __init__(self):
        self._q      = queue.Queue()   # thread-safe event queue
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._root   = None

        # Match state
        self.score_a    = 0
        self.score_b    = 0
        self.sets_a     = 0
        self.sets_b     = 0
        self.set_history = []          # [(a_score, b_score), …]
        self.ch_a        = self.CHALLENGES_PER_SET
        self.ch_b        = self.CHALLENGES_PER_SET
        self.serve       = "A"         # "A" or "B"
        self.rally_log   = []          # list of log-entry dicts
        self.game_over   = False
        self.set_over    = False
        self.name_a      = "Player A"
        self.name_b      = "Player B"
        self._start_time = time.time()
        self._last_decision = None     # (decision, cm, px)
        self._set_num    = 1

    # ═══════════════════════════════════════════════════════════════
    #  Public API (thread-safe — safe to call from CV loop)
    # ═══════════════════════════════════════════════════════════════

    def start(self):
        """Start the dashboard window in a background thread."""
        self._thread.start()

    def stop(self):
        """Close the dashboard."""
        self._q.put(("QUIT", None))

    def push_decision(self, decision, cm, px):
        """
        Called by line_judge callback or app loop.
        decision: "IN" or "OUT"
        cm: (rx, ry) real-world centimetres
        px: (px, py) pixel coordinates
        """
        self._q.put(("DECISION", (decision, cm, px)))

    def add_point(self, side):
        """Manually add a point to side 'A' or 'B'."""
        self._q.put(("POINT", side))

    # ═══════════════════════════════════════════════════════════════
    #  Internal thread entry
    # ═══════════════════════════════════════════════════════════════

    def _run(self):
        self._root = tk.Tk()
        self._root.title("ShuttleEye  ─  Umpire Dashboard")
        self._root.configure(bg=BG)
        self._root.geometry("1000x760")
        self._root.resizable(True, True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._poll_queue()
        self._tick_timer()
        self._root.mainloop()

    def _on_close(self):
        self._root.destroy()

    # ═══════════════════════════════════════════════════════════════
    #  UI construction
    # ═══════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = self._root

        # ── Top title bar ─────────────────────────────────────────
        title_bar = tk.Frame(root, bg=BG, pady=6)
        title_bar.pack(fill="x")

        tk.Label(title_bar, text="🏸  ShuttleEye Umpire Dashboard",
                 font=("Helvetica", 16, "bold"),
                 fg=CYAN, bg=BG).pack(side="left", padx=16)

        self._timer_var = tk.StringVar(value="00:00")
        tk.Label(title_bar, textvariable=self._timer_var,
                 font=("Courier", 14, "bold"),
                 fg=TEXT_DIM, bg=BG).pack(side="right", padx=16)

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        # ── Main body split ───────────────────────────────────────
        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        left  = tk.Frame(body, bg=BG, width=540)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        left.pack_propagate(False)

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        right = tk.Frame(body, bg=BG, width=450)
        right.pack(side="right", fill="both", padx=8, pady=8)
        right.pack_propagate(False)

        self._build_left(left)
        self._build_right(right)

    # ── Left panel: scores + controls ─────────────────────────────

    def _build_left(self, parent):

        # Player name row
        name_row = tk.Frame(parent, bg=BG)
        name_row.pack(fill="x", pady=(4, 0))

        tk.Label(name_row, text="Player / Team names:", fg=TEXT_DIM,
                 bg=BG, font=("Helvetica", 10)).pack(side="left")
        tk.Button(name_row, text="✏  Edit", command=self._edit_names,
                  fg=CYAN, bg=BG3, bd=0, padx=8, pady=2,
                  font=("Helvetica", 9)).pack(side="right")

        # ── Scoreboard ────────────────────────────────────────────
        sb = tk.Frame(parent, bg=BG2, bd=0, relief="flat")
        sb.pack(fill="x", pady=6)
        sb.columnconfigure(0, weight=1)
        sb.columnconfigure(1, weight=0)
        sb.columnconfigure(2, weight=1)

        # Name labels
        self._name_a_var = tk.StringVar(value=self.name_a)
        self._name_b_var = tk.StringVar(value=self.name_b)

        tk.Label(sb, textvariable=self._name_a_var,
                 fg=SCORE_A_COL, bg=BG2,
                 font=("Helvetica", 15, "bold")).grid(row=0, column=0, pady=(12,0))
        tk.Label(sb, text="vs",
                 fg=TEXT_DIM, bg=BG2,
                 font=("Helvetica", 12)).grid(row=0, column=1, pady=(12,0))
        tk.Label(sb, textvariable=self._name_b_var,
                 fg=SCORE_B_COL, bg=BG2,
                 font=("Helvetica", 15, "bold")).grid(row=0, column=2, pady=(12,0))

        # Big score numbers
        self._score_a_var = tk.StringVar(value="0")
        self._score_b_var = tk.StringVar(value="0")

        self._score_a_label = tk.Label(sb, textvariable=self._score_a_var,
                 fg=SCORE_A_COL, bg=BG2,
                 font=("Helvetica", 80, "bold"))
        self._score_a_label.grid(row=1, column=0, padx=20, pady=4)

        tk.Label(sb, text="—", fg=TEXT_DIM, bg=BG2,
                 font=("Helvetica", 36)).grid(row=1, column=1)

        self._score_b_label = tk.Label(sb, textvariable=self._score_b_var,
                 fg=SCORE_B_COL, bg=BG2,
                 font=("Helvetica", 80, "bold"))
        self._score_b_label.grid(row=1, column=2, padx=20, pady=4)

        # Serve indicator
        self._serve_var = tk.StringVar(value=f"🏸  Serving: {self.name_a}")
        tk.Label(sb, textvariable=self._serve_var,
                 fg=YELLOW, bg=BG2,
                 font=("Helvetica", 11)).grid(row=2, column=0, columnspan=3, pady=(0,4))

        # Deuce / status
        self._status_var = tk.StringVar(value="")
        self._status_lbl = tk.Label(sb, textvariable=self._status_var,
                 fg=YELLOW, bg=BG2,
                 font=("Helvetica", 13, "bold"))
        self._status_lbl.grid(row=3, column=0, columnspan=3, pady=(0, 8))

        # ── Set history ───────────────────────────────────────────
        set_frame = tk.Frame(parent, bg=BG3, bd=0)
        set_frame.pack(fill="x", pady=4, ipady=6, ipadx=8)

        tk.Label(set_frame, text="SET HISTORY",
                 fg=TEXT_DIM, bg=BG3,
                 font=("Helvetica", 9, "bold")).pack(anchor="w", padx=8)

        self._sets_var = tk.StringVar(value="–")
        tk.Label(set_frame, textvariable=self._sets_var,
                 fg=TEXT, bg=BG3,
                 font=("Courier", 11)).pack(anchor="w", padx=8)

        # Sets won row
        sw = tk.Frame(parent, bg=BG)
        sw.pack(fill="x", pady=2)

        tk.Label(sw, text="Sets won:", fg=TEXT_DIM, bg=BG,
                 font=("Helvetica", 10)).pack(side="left")
        self._sets_won_var = tk.StringVar(value="A: 0   B: 0")
        tk.Label(sw, textvariable=self._sets_won_var,
                 fg=TEXT, bg=BG, font=("Helvetica", 10, "bold")).pack(side="left", padx=8)

        # ── Challenge counters ────────────────────────────────────
        ch_frame = tk.Frame(parent, bg=BG3)
        ch_frame.pack(fill="x", pady=4, ipady=6)

        tk.Label(ch_frame, text="CHALLENGES REMAINING",
                 fg=TEXT_DIM, bg=BG3,
                 font=("Helvetica", 9, "bold")).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(2,4))

        self._ch_a_var = tk.StringVar(value="●●")
        self._ch_b_var = tk.StringVar(value="●●")

        tk.Label(ch_frame, textvariable=self._name_a_var,
                 fg=SCORE_A_COL, bg=BG3, font=("Helvetica", 10)).grid(row=1, column=0, padx=8)
        tk.Label(ch_frame, textvariable=self._ch_a_var,
                 fg=GREEN, bg=BG3, font=("Helvetica", 16)).grid(row=1, column=1, padx=4)
        tk.Label(ch_frame, textvariable=self._name_b_var,
                 fg=SCORE_B_COL, bg=BG3, font=("Helvetica", 10)).grid(row=1, column=2, padx=8)
        tk.Label(ch_frame, textvariable=self._ch_b_var,
                 fg=GREEN, bg=BG3, font=("Helvetica", 16)).grid(row=1, column=3, padx=4)

        ch_frame.columnconfigure(0, weight=1)
        ch_frame.columnconfigure(1, weight=1)
        ch_frame.columnconfigure(2, weight=1)
        ch_frame.columnconfigure(3, weight=1)

        # ── Decision flash ────────────────────────────────────────
        self._decision_var = tk.StringVar(value="")
        self._decision_lbl = tk.Label(parent, textvariable=self._decision_var,
                 font=("Helvetica", 28, "bold"),
                 bg=BG, fg=GREEN, pady=6)
        self._decision_lbl.pack(fill="x")

        # ── Control buttons ───────────────────────────────────────
        self._build_controls(parent)

    def _build_controls(self, parent):
        tk.Label(parent, text="UMPIRE CONTROLS",
                 fg=TEXT_DIM, bg=BG,
                 font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(8, 2))

        ctrl = tk.Frame(parent, bg=BG)
        ctrl.pack(fill="x")

        # Point buttons
        row1 = tk.Frame(ctrl, bg=BG);  row1.pack(fill="x", pady=2)
        self._btn(row1, f"＋ Point A", lambda: self.add_point("A"),
                  SCORE_A_COL, BG3).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row1, f"＋ Point B", lambda: self.add_point("B"),
                  SCORE_B_COL, BG3).pack(side="left", expand=True, fill="x", padx=2)

        # Override last call
        row2 = tk.Frame(ctrl, bg=BG);  row2.pack(fill="x", pady=2)
        self._btn(row2, "✓ Override → IN", self._override_in,
                  GREEN, BG3).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row2, "✗ Override → OUT", self._override_out,
                  RED, BG3).pack(side="left", expand=True, fill="x", padx=2)

        # Challenge buttons
        row3 = tk.Frame(ctrl, bg=BG);  row3.pack(fill="x", pady=2)
        self._btn(row3, "🏳 Challenge A", lambda: self._challenge("A"),
                  YELLOW, BG3).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row3, "🏳 Challenge B", lambda: self._challenge("B"),
                  YELLOW, BG3).pack(side="left", expand=True, fill="x", padx=2)

        # Serve toggle / undo / reset
        row4 = tk.Frame(ctrl, bg=BG);  row4.pack(fill="x", pady=2)
        self._btn(row4, "🔄 Toggle Serve", self._toggle_serve,
                  TEXT_DIM, BG3).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row4, "↩ Undo Last Pt", self._undo_point,
                  ORANGE, BG3).pack(side="left", expand=True, fill="x", padx=2)

        row5 = tk.Frame(ctrl, bg=BG);  row5.pack(fill="x", pady=2)
        self._btn(row5, "🔁 New Set", self._new_set,
                  CYAN, BG3).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row5, "🗑 Reset Match", self._reset_match,
                  RED, BG3).pack(side="left", expand=True, fill="x", padx=2)

        row6 = tk.Frame(ctrl, bg=BG);  row6.pack(fill="x", pady=2)
        self._btn(row6, "💾 Export Log", self._export_log,
                  TEXT_DIM, BG3).pack(side="left", expand=True, fill="x", padx=2)

    # ── Right panel: rally log ────────────────────────────────────

    def _build_right(self, parent):
        tk.Label(parent, text="RALLY LOG",
                 fg=TEXT_DIM, bg=BG,
                 font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(4, 2))

        log_frame = tk.Frame(parent, bg=BG2, bd=1, relief="solid")
        log_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame, bg=BG3, troughcolor=BG2)
        scrollbar.pack(side="right", fill="y")

        self._log_list = tk.Listbox(
            log_frame,
            yscrollcommand=scrollbar.set,
            bg=BG2, fg=TEXT,
            font=("Courier", 10),
            selectbackground=BG3,
            selectforeground=WHITE,
            bd=0,
            highlightthickness=0,
            activestyle="none",
        )
        self._log_list.pack(fill="both", expand=True)
        scrollbar.config(command=self._log_list.yview)

        # Summary row at bottom of right panel
        sum_frame = tk.Frame(parent, bg=BG3)
        sum_frame.pack(fill="x", pady=(4,0), ipady=4)

        tk.Label(sum_frame, text="Session stats:",
                 fg=TEXT_DIM, bg=BG3, font=("Helvetica", 9)).pack(side="left", padx=6)
        self._stats_var = tk.StringVar(value="Rallies: 0  |  IN: 0  |  OUT: 0")
        tk.Label(sum_frame, textvariable=self._stats_var,
                 fg=TEXT, bg=BG3, font=("Courier", 9)).pack(side="left")

    # ── Widget helper ─────────────────────────────────────────────

    def _btn(self, parent, text, cmd, fg, bg):
        return tk.Button(parent, text=text, command=cmd,
                         fg=fg, bg=bg, bd=0,
                         font=("Helvetica", 10, "bold"),
                         padx=6, pady=5,
                         activebackground=BORDER,
                         activeforeground=WHITE,
                         cursor="hand2")

    # ═══════════════════════════════════════════════════════════════
    #  Queue polling (runs on Tkinter thread via after())
    # ═══════════════════════════════════════════════════════════════

    def _poll_queue(self):
        try:
            while True:
                event, data = self._q.get_nowait()
                if event == "QUIT":
                    self._root.destroy();  return
                elif event == "DECISION":
                    decision, cm, px = data
                    self._handle_decision(decision, cm, px)
                elif event == "POINT":
                    self._apply_point(data)
        except queue.Empty:
            pass
        self._root.after(50, self._poll_queue)

    # ═══════════════════════════════════════════════════════════════
    #  Core logic
    # ═══════════════════════════════════════════════════════════════

    def _handle_decision(self, decision, cm, px):
        """Process a line-judge IN/OUT decision."""
        self._last_decision = (decision, cm, px)

        # Auto-assign point based on decision
        # IN  → receiving side (B) loses rally → A scores
        # OUT → serving side's shuttle → B scores
        # (umpire can always override)
        point_side = "A" if decision == "IN" else "B"
        self._apply_point(point_side, decision=decision, cm=cm)

    def _apply_point(self, side, decision=None, cm=None, note=""):
        if self.game_over:
            return

        if side == "A":
            self.score_a += 1
            self.serve    = "A"
        else:
            self.score_b += 1
            self.serve    = "B"

        # Undo stack entry
        self._undo_stack_push()

        # Log entry
        rally_num = len(self.rally_log) + 1
        ts        = datetime.datetime.now().strftime("%H:%M:%S")
        cm_str    = f"({cm[0]:.0f},{cm[1]:.0f})cm" if cm else ""
        dec_str   = decision if decision else "manual"
        entry = {
            "num"     : rally_num,
            "ts"      : ts,
            "decision": dec_str,
            "side"    : side,
            "cm"      : cm_str,
            "score_a" : self.score_a,
            "score_b" : self.score_b,
            "note"    : note,
            "set"     : self._set_num,
        }
        self.rally_log.append(entry)
        self._add_log_row(entry)

        self._check_set_over()
        self._refresh_ui()

    def _check_set_over(self):
        a, b = self.score_a, self.score_b
        won  = False
        if a >= self.WINNING_SCORE or b >= self.WINNING_SCORE:
            if abs(a-b) >= 2 or max(a,b) >= self.MAX_SCORE:
                won = True

        if won:
            winner = "A" if a > b else "B"
            if winner == "A":
                self.sets_a += 1
            else:
                self.sets_b += 1
            self.set_history.append((a, b))

            # Check match winner
            if self.sets_a > self.BEST_OF // 2 or self.sets_b > self.BEST_OF // 2:
                self.game_over = True
                match_winner   = self.name_a if self.sets_a > self.sets_b else self.name_b
                self._status_var.set(f"🏆  {match_winner} WINS THE MATCH!")
                self._log_separator(f"MATCH WON BY {match_winner}")
            else:
                self._status_var.set(f"Set {self._set_num} over  —  Press 'New Set' to continue")
                self._log_separator(f"SET {self._set_num} END  {a}–{b}")
                self.set_over = True

    def _new_set(self):
        if not self.set_over and not self.game_over:
            if not messagebox.askyesno("New Set", "Current set not finished. Start new set anyway?"):
                return
        self._set_num  += 1
        self.score_a    = 0
        self.score_b    = 0
        self.ch_a       = self.CHALLENGES_PER_SET
        self.ch_b       = self.CHALLENGES_PER_SET
        self.set_over   = False
        self._status_var.set(f"Set {self._set_num} started")
        self._log_separator(f"─── SET {self._set_num} ───")
        self._undo_stack = []
        self._refresh_ui()

    def _reset_match(self):
        if not messagebox.askyesno("Reset Match", "Reset all scores and rally log?"):
            return
        self.score_a    = 0
        self.score_b    = 0
        self.sets_a     = 0
        self.sets_b     = 0
        self.set_history= []
        self.ch_a       = self.CHALLENGES_PER_SET
        self.ch_b       = self.CHALLENGES_PER_SET
        self.serve      = "A"
        self.game_over  = False
        self.set_over   = False
        self._set_num   = 1
        self.rally_log  = []
        self._start_time= time.time()
        self._status_var.set("")
        self._undo_stack= []
        self._log_list.delete(0, tk.END)
        self._refresh_ui()

    # ── Override / challenge ──────────────────────────────────────

    def _override_in(self):
        if self._last_decision:
            dec, cm, px = self._last_decision
            if dec == "OUT":
                # Reverse: remove B's point, give A a point
                self._undo_point(silent=True)
                self._apply_point("A", decision="IN (override)", cm=cm,
                                  note="umpire override")
                self._flash_decision("OVERRIDE → IN", GREEN)

    def _override_out(self):
        if self._last_decision:
            dec, cm, px = self._last_decision
            if dec == "IN":
                self._undo_point(silent=True)
                self._apply_point("B", decision="OUT (override)", cm=cm,
                                  note="umpire override")
                self._flash_decision("OVERRIDE → OUT", RED)

    def _challenge(self, side):
        if self.game_over:
            return
        ch = self.ch_a if side == "A" else self.ch_b
        if ch <= 0:
            self._flash_decision(f"No challenges left for {side}", RED)
            return

        # Determine if challenge is successful (uses last decision)
        if not self._last_decision:
            self._flash_decision("No decision to challenge", RED)
            return

        result = messagebox.askyesno(
            f"Challenge by {getattr(self, 'name_'+side.lower())}",
            "Was the challenge SUCCESSFUL?\n\n"
            "Yes = call was wrong (challenger keeps challenge if original call reversed)\n"
            "No = call was correct (challenger loses challenge)"
        )

        if result:
            # Successful: don't consume challenge, reverse call
            self._flash_decision(f"Challenge {side} SUCCESSFUL", GREEN)
            self._log_separator(f"✓ CHALLENGE BY {side} — CALL REVERSED")
            if side == "A":
                self._override_in() if self._last_decision[0] == "OUT" else self._override_out()
            else:
                self._override_out() if self._last_decision[0] == "IN" else self._override_in()
        else:
            # Failed: consume challenge
            if side == "A":
                self.ch_a = max(0, self.ch_a - 1)
            else:
                self.ch_b = max(0, self.ch_b - 1)
            self._flash_decision(f"Challenge {side} FAILED", RED)
            self._log_separator(f"✗ CHALLENGE BY {side} — CALL STANDS")

        self._refresh_ui()

    # ── Undo ─────────────────────────────────────────────────────

    _undo_stack = []

    def _undo_stack_push(self):
        self._undo_stack.append((self.score_a, self.score_b, self.serve))

    def _undo_point(self, silent=False):
        if not self._undo_stack:
            if not silent:
                self._flash_decision("Nothing to undo", RED)
            return
        self.score_a, self.score_b, self.serve = self._undo_stack.pop()
        if self.rally_log:
            self.rally_log.pop()
            self._log_list.delete(tk.END)
        self.set_over   = False
        self.game_over  = False
        self._status_var.set("")
        self._refresh_ui()

    # ── Serve ─────────────────────────────────────────────────────

    def _toggle_serve(self):
        self.serve = "B" if self.serve == "A" else "A"
        self._refresh_ui()

    # ── Name editor ───────────────────────────────────────────────

    def _edit_names(self):
        win = tk.Toplevel(self._root)
        win.title("Edit Names")
        win.configure(bg=BG)
        win.geometry("360x160")
        win.resizable(False, False)

        tk.Label(win, text="Player A name:", fg=TEXT, bg=BG,
                 font=("Helvetica", 11)).grid(row=0, column=0, padx=12, pady=8, sticky="w")
        ea = tk.Entry(win, font=("Helvetica", 11), bg=BG3, fg=WHITE,
                      insertbackground=WHITE, bd=1)
        ea.insert(0, self.name_a)
        ea.grid(row=0, column=1, padx=8, pady=8)

        tk.Label(win, text="Player B name:", fg=TEXT, bg=BG,
                 font=("Helvetica", 11)).grid(row=1, column=0, padx=12, pady=4, sticky="w")
        eb = tk.Entry(win, font=("Helvetica", 11), bg=BG3, fg=WHITE,
                      insertbackground=WHITE, bd=1)
        eb.insert(0, self.name_b)
        eb.grid(row=1, column=1, padx=8, pady=4)

        def _apply():
            self.name_a = ea.get().strip() or "Player A"
            self.name_b = eb.get().strip() or "Player B"
            self._name_a_var.set(self.name_a)
            self._name_b_var.set(self.name_b)
            self._refresh_ui()
            win.destroy()

        tk.Button(win, text="Save", command=_apply,
                  fg=WHITE, bg=CYAN, bd=0, padx=12, pady=4,
                  font=("Helvetica", 11, "bold")).grid(
                  row=2, column=0, columnspan=2, pady=10)

    # ── Export ────────────────────────────────────────────────────

    def _export_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"shuttleeye_rally_log_{datetime.date.today()}.txt"
        )
        if not path:
            return
        lines = [
            "ShuttleEye — Umpire Rally Log",
            f"Date: {datetime.date.today()}",
            f"Players: {self.name_a} vs {self.name_b}",
            f"Sets won: A={self.sets_a}  B={self.sets_b}",
            "=" * 60,
            ""
        ]
        for e in self.rally_log:
            line = (f"[{e['ts']}] Rally {e['num']:3d}  Set {e['set']}  "
                    f"{e['decision']:12s}  Pt→{e['side']}  "
                    f"Score {e['score_a']}–{e['score_b']}  {e['cm']}")
            if e['note']:
                line += f"  [{e['note']}]"
            lines.append(line)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self._flash_decision(f"Log saved", GREEN)

    # ═══════════════════════════════════════════════════════════════
    #  UI refresh
    # ═══════════════════════════════════════════════════════════════

    def _refresh_ui(self):
        a, b = self.score_a, self.score_b
        self._score_a_var.set(str(a))
        self._score_b_var.set(str(b))

        # Deuce / advantage status
        if a >= self.DEUCE_SCORE and b >= self.DEUCE_SCORE:
            if a == b:
                self._status_var.set("DEUCE")
            elif a > b:
                self._status_var.set(f"ADVANTAGE  {self.name_a}")
            else:
                self._status_var.set(f"ADVANTAGE  {self.name_b}")
        elif not self.set_over and not self.game_over:
            self._status_var.set("")

        # Score label flash colour
        self._score_a_label.config(
            fg=WHITE if self.serve=="A" else SCORE_A_COL)
        self._score_b_label.config(
            fg=WHITE if self.serve=="B" else SCORE_B_COL)

        # Serve
        sn = self.name_a if self.serve=="A" else self.name_b
        self._serve_var.set(f"🏸  Serving: {sn}")

        # Challenges
        self._ch_a_var.set("●" * self.ch_a + "○" * (self.CHALLENGES_PER_SET - self.ch_a))
        self._ch_b_var.set("●" * self.ch_b + "○" * (self.CHALLENGES_PER_SET - self.ch_b))

        # Set history
        if self.set_history:
            history_strs = [f"Set{i+1}: {x}–{y}" for i,(x,y) in enumerate(self.set_history)]
            self._sets_var.set("  ".join(history_strs))
        else:
            self._sets_var.set(f"Set {self._set_num} in progress")

        # Sets won
        self._sets_won_var.set(
            f"{self.name_a}: {self.sets_a}   {self.name_b}: {self.sets_b}")

        # Stats
        total = len(self.rally_log)
        ins   = sum(1 for e in self.rally_log if e['decision'].startswith("IN"))
        outs  = total - ins
        self._stats_var.set(f"Rallies: {total}  |  IN: {ins}  |  OUT: {outs}")

    def _add_log_row(self, entry):
        a, b  = entry['score_a'], entry['score_b']
        dec   = entry['decision']
        side  = entry['side']
        ts    = entry['ts']
        cm    = entry['cm']
        note  = f" [{entry['note']}]" if entry['note'] else ""
        text  = (f"#{entry['num']:3d} {ts}  {dec:<14s} Pt→{side}  "
                 f"{a}–{b}  {cm}{note}")
        self._log_list.insert(tk.END, text)

        # Colour tag
        idx  = self._log_list.size() - 1
        col  = GREEN if dec.startswith("IN") else RED
        self._log_list.itemconfig(idx, fg=col)
        self._log_list.see(tk.END)

    def _log_separator(self, text=""):
        line = f"── {text} {'─'*max(0, 42-len(text))}"
        self._log_list.insert(tk.END, line)
        self._log_list.itemconfig(self._log_list.size()-1, fg=TEXT_DIM)
        self._log_list.see(tk.END)

    def _flash_decision(self, text, color):
        self._decision_var.set(text)
        self._decision_lbl.config(fg=color)
        # Clear after 2.5 s
        self._root.after(2500, lambda: self._decision_var.set(""))

    # ═══════════════════════════════════════════════════════════════
    #  Timer
    # ═══════════════════════════════════════════════════════════════

    def _tick_timer(self):
        elapsed = int(time.time() - self._start_time)
        m, s    = divmod(elapsed, 60)
        self._timer_var.set(f"{m:02d}:{s:02d}")
        self._root.after(1000, self._tick_timer)
