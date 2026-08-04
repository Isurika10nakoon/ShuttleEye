# umpire_dashboard.py  ─  ShuttleEye v5  ─  Umpire Scoring Dashboard
# ═══════════════════════════════════════════════════════════════════════
#
#  A professional CustomTkinter umpire dashboard that:
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
#  • Shows which logged-in umpire is running the match
#  • Runs in its own thread — non-blocking to the CV loop
#
#  USAGE
#  ─────
#  from umpire_dashboard import UmpireDashboard
#  dash = UmpireDashboard(umpire_name="Jane Doe", role="umpire")
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
from tkinter import messagebox, filedialog
import customtkinter as ctk
import threading
import time
import datetime
import queue

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ── Colour palette ────────────────────────────────────────────────────
BG          = "#0d1117"
BG2         = "#161b22"
BG3         = "#21262d"
BG4         = "#282e37"
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

FONT_FAMILY = "Segoe UI"


class UmpireDashboard:
    """
    Umpire scoring dashboard.
    Runs CustomTkinter in a dedicated daemon thread so it never blocks the CV loop.
    """

    WINNING_SCORE  = 21
    DEUCE_SCORE    = 20
    MAX_SCORE      = 30
    BEST_OF        = 3          # match is best of 3 sets
    CHALLENGES_PER_SET = 2

    def __init__(self, umpire_name="Umpire", role="umpire"):
        self._q      = queue.Queue()   # thread-safe event queue
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._root   = None

        # Session / login info
        self.umpire_name = umpire_name
        self.role        = role

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
        cm: (perp_dist_px, along_line_px) — signed pixel offsets from the
            boundary line (named "cm" for historical/interface reasons;
            there's no real-world scale for a single line)
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
        self._root = ctk.CTk()
        self._root.title("ShuttleEye  ─  Umpire Dashboard")
        self._root.configure(fg_color=BG)
        self._root.geometry("1040x780")
        self._root.minsize(920, 680)
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
        title_bar = ctk.CTkFrame(root, fg_color=BG, corner_radius=0, height=52)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        ctk.CTkLabel(title_bar, text="🏸  ShuttleEye Umpire Dashboard",
                     font=(FONT_FAMILY, 18, "bold"),
                     text_color=CYAN, fg_color="transparent").pack(side="left", padx=18)

        role_badge = "ADMIN" if self.role == "admin" else "UMPIRE"
        badge_col  = YELLOW if self.role == "admin" else GREEN
        user_frame = ctk.CTkFrame(title_bar, fg_color=BG3, corner_radius=8)
        user_frame.pack(side="right", padx=16, pady=8)
        ctk.CTkLabel(user_frame, text=f"👤 {self.umpire_name}",
                     font=(FONT_FAMILY, 12, "bold"), text_color=TEXT,
                     fg_color="transparent").pack(side="left", padx=(12, 6), pady=4)
        ctk.CTkLabel(user_frame, text=role_badge,
                     font=(FONT_FAMILY, 10, "bold"), text_color=badge_col,
                     fg_color="transparent").pack(side="left", padx=(0, 12), pady=4)

        self._timer_var = tk.StringVar(value="00:00")
        ctk.CTkLabel(title_bar, textvariable=self._timer_var,
                     font=("Consolas", 15, "bold"),
                     text_color=TEXT_DIM, fg_color="transparent").pack(side="right", padx=8)

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        # ── Main body split ───────────────────────────────────────
        body = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        left  = ctk.CTkFrame(body, fg_color=BG, corner_radius=0, width=560)
        left.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        left.pack_propagate(False)

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        right = ctk.CTkFrame(body, fg_color=BG, corner_radius=0, width=440)
        right.pack(side="right", fill="both", padx=10, pady=10)
        right.pack_propagate(False)

        self._build_left(left)
        self._build_right(right)

    # ── Left panel: scores + controls ─────────────────────────────

    def _build_left(self, parent):

        # Player name row
        name_row = ctk.CTkFrame(parent, fg_color="transparent")
        name_row.pack(fill="x", pady=(4, 0))

        ctk.CTkLabel(name_row, text="Player / Team names:", text_color=TEXT_DIM,
                     fg_color="transparent", font=(FONT_FAMILY, 11)).pack(side="left")
        ctk.CTkButton(name_row, text="✏  Edit", command=self._edit_names,
                      text_color=CYAN, fg_color=BG3, hover_color=BG4,
                      corner_radius=8, width=70, height=26,
                      font=(FONT_FAMILY, 10)).pack(side="right")

        # ── Scoreboard ────────────────────────────────────────────
        sb = ctk.CTkFrame(parent, fg_color=BG2, corner_radius=14)
        sb.pack(fill="x", pady=8)
        sb.columnconfigure(0, weight=1)
        sb.columnconfigure(1, weight=0)
        sb.columnconfigure(2, weight=1)

        # Name labels
        self._name_a_var = tk.StringVar(value=self.name_a)
        self._name_b_var = tk.StringVar(value=self.name_b)

        ctk.CTkLabel(sb, textvariable=self._name_a_var,
                     text_color=SCORE_A_COL, fg_color="transparent",
                     font=(FONT_FAMILY, 16, "bold")).grid(row=0, column=0, pady=(16, 0))
        ctk.CTkLabel(sb, text="vs",
                     text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 12)).grid(row=0, column=1, pady=(16, 0))
        ctk.CTkLabel(sb, textvariable=self._name_b_var,
                     text_color=SCORE_B_COL, fg_color="transparent",
                     font=(FONT_FAMILY, 16, "bold")).grid(row=0, column=2, pady=(16, 0))

        # Big score numbers
        self._score_a_var = tk.StringVar(value="0")
        self._score_b_var = tk.StringVar(value="0")

        self._score_a_label = ctk.CTkLabel(sb, textvariable=self._score_a_var,
                 text_color=SCORE_A_COL, fg_color="transparent",
                 font=(FONT_FAMILY, 84, "bold"))
        self._score_a_label.grid(row=1, column=0, padx=20, pady=6)

        ctk.CTkLabel(sb, text="—", text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 36)).grid(row=1, column=1)

        self._score_b_label = ctk.CTkLabel(sb, textvariable=self._score_b_var,
                 text_color=SCORE_B_COL, fg_color="transparent",
                 font=(FONT_FAMILY, 84, "bold"))
        self._score_b_label.grid(row=1, column=2, padx=20, pady=6)

        # Serve indicator
        self._serve_var = tk.StringVar(value=f"🏸  Serving: {self.name_a}")
        ctk.CTkLabel(sb, textvariable=self._serve_var,
                     text_color=YELLOW, fg_color="transparent",
                     font=(FONT_FAMILY, 12)).grid(row=2, column=0, columnspan=3, pady=(0, 4))

        # Deuce / status
        self._status_var = tk.StringVar(value="")
        self._status_lbl = ctk.CTkLabel(sb, textvariable=self._status_var,
                 text_color=YELLOW, fg_color="transparent",
                 font=(FONT_FAMILY, 14, "bold"))
        self._status_lbl.grid(row=3, column=0, columnspan=3, pady=(0, 12))

        # ── Set history ───────────────────────────────────────────
        set_frame = ctk.CTkFrame(parent, fg_color=BG3, corner_radius=12)
        set_frame.pack(fill="x", pady=5, ipady=6)

        ctk.CTkLabel(set_frame, text="SET HISTORY",
                     text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=12, pady=(6, 0))

        self._sets_var = tk.StringVar(value="–")
        ctk.CTkLabel(set_frame, textvariable=self._sets_var,
                     text_color=TEXT, fg_color="transparent",
                     font=("Consolas", 12)).pack(anchor="w", padx=12, pady=(0, 4))

        # Sets won row
        sw = ctk.CTkFrame(parent, fg_color="transparent")
        sw.pack(fill="x", pady=3)

        ctk.CTkLabel(sw, text="Sets won:", text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 11)).pack(side="left")
        self._sets_won_var = tk.StringVar(value="A: 0   B: 0")
        ctk.CTkLabel(sw, textvariable=self._sets_won_var,
                     text_color=TEXT, fg_color="transparent",
                     font=(FONT_FAMILY, 11, "bold")).pack(side="left", padx=8)

        # ── Challenge counters ────────────────────────────────────
        ch_frame = ctk.CTkFrame(parent, fg_color=BG3, corner_radius=12)
        ch_frame.pack(fill="x", pady=5, ipady=6)

        ctk.CTkLabel(ch_frame, text="CHALLENGES REMAINING",
                     text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 10, "bold")).grid(row=0, column=0, columnspan=4,
                                                           sticky="w", padx=12, pady=(6, 4))

        self._ch_a_var = tk.StringVar(value="●●")
        self._ch_b_var = tk.StringVar(value="●●")

        ctk.CTkLabel(ch_frame, textvariable=self._name_a_var,
                     text_color=SCORE_A_COL, fg_color="transparent",
                     font=(FONT_FAMILY, 11)).grid(row=1, column=0, padx=8, pady=(0, 6))
        ctk.CTkLabel(ch_frame, textvariable=self._ch_a_var,
                     text_color=GREEN, fg_color="transparent",
                     font=(FONT_FAMILY, 17)).grid(row=1, column=1, padx=4, pady=(0, 6))
        ctk.CTkLabel(ch_frame, textvariable=self._name_b_var,
                     text_color=SCORE_B_COL, fg_color="transparent",
                     font=(FONT_FAMILY, 11)).grid(row=1, column=2, padx=8, pady=(0, 6))
        ctk.CTkLabel(ch_frame, textvariable=self._ch_b_var,
                     text_color=GREEN, fg_color="transparent",
                     font=(FONT_FAMILY, 17)).grid(row=1, column=3, padx=4, pady=(0, 6))

        ch_frame.columnconfigure(0, weight=1)
        ch_frame.columnconfigure(1, weight=1)
        ch_frame.columnconfigure(2, weight=1)
        ch_frame.columnconfigure(3, weight=1)

        # ── Decision flash ────────────────────────────────────────
        self._decision_var = tk.StringVar(value="")
        self._decision_lbl = ctk.CTkLabel(parent, textvariable=self._decision_var,
                 font=(FONT_FAMILY, 26, "bold"),
                 fg_color="transparent", text_color=GREEN)
        self._decision_lbl.pack(fill="x", pady=4)

        # ── Control buttons ───────────────────────────────────────
        self._build_controls(parent)

    def _build_controls(self, parent):
        ctk.CTkLabel(parent, text="UMPIRE CONTROLS",
                     text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(6, 4))

        ctrl = ctk.CTkFrame(parent, fg_color="transparent")
        ctrl.pack(fill="x")

        # Point buttons
        row1 = ctk.CTkFrame(ctrl, fg_color="transparent");  row1.pack(fill="x", pady=3)
        self._btn(row1, "＋ Point A", lambda: self.add_point("A"),
                  SCORE_A_COL).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row1, "＋ Point B", lambda: self.add_point("B"),
                  SCORE_B_COL).pack(side="left", expand=True, fill="x", padx=2)

        # Override last call
        row2 = ctk.CTkFrame(ctrl, fg_color="transparent");  row2.pack(fill="x", pady=3)
        self._btn(row2, "✓ Override → IN", self._override_in,
                  GREEN).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row2, "✗ Override → OUT", self._override_out,
                  RED).pack(side="left", expand=True, fill="x", padx=2)

        # Challenge buttons
        row3 = ctk.CTkFrame(ctrl, fg_color="transparent");  row3.pack(fill="x", pady=3)
        self._btn(row3, "🏳 Challenge A", lambda: self._challenge("A"),
                  YELLOW).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row3, "🏳 Challenge B", lambda: self._challenge("B"),
                  YELLOW).pack(side="left", expand=True, fill="x", padx=2)

        # Serve toggle / undo / reset
        row4 = ctk.CTkFrame(ctrl, fg_color="transparent");  row4.pack(fill="x", pady=3)
        self._btn(row4, "🔄 Toggle Serve", self._toggle_serve,
                  TEXT_DIM).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row4, "↩ Undo Last Pt", self._undo_point,
                  ORANGE).pack(side="left", expand=True, fill="x", padx=2)

        row5 = ctk.CTkFrame(ctrl, fg_color="transparent");  row5.pack(fill="x", pady=3)
        self._btn(row5, "🔁 New Set", self._new_set,
                  CYAN).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row5, "🗑 Reset Match", self._reset_match,
                  RED).pack(side="left", expand=True, fill="x", padx=2)

        row6 = ctk.CTkFrame(ctrl, fg_color="transparent");  row6.pack(fill="x", pady=3)
        self._btn(row6, "💾 Export Log", self._export_log,
                  TEXT_DIM).pack(side="left", expand=True, fill="x", padx=2)

    # ── Right panel: rally log ────────────────────────────────────

    def _build_right(self, parent):
        ctk.CTkLabel(parent, text="RALLY LOG",
                     text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(4, 4))

        log_frame = ctk.CTkFrame(parent, fg_color=BG2, corner_radius=12,
                                  border_width=1, border_color=BORDER)
        log_frame.pack(fill="both", expand=True)

        inner = tk.Frame(log_frame, bg=BG2)
        inner.pack(fill="both", expand=True, padx=8, pady=8)

        scrollbar = tk.Scrollbar(inner, bg=BG3, troughcolor=BG2)
        scrollbar.pack(side="right", fill="y")

        self._log_list = tk.Listbox(
            inner,
            yscrollcommand=scrollbar.set,
            bg=BG2, fg=TEXT,
            font=("Consolas", 10),
            selectbackground=BG4,
            selectforeground=WHITE,
            bd=0,
            highlightthickness=0,
            activestyle="none",
        )
        self._log_list.pack(fill="both", expand=True)
        scrollbar.config(command=self._log_list.yview)

        # Summary row at bottom of right panel
        sum_frame = ctk.CTkFrame(parent, fg_color=BG3, corner_radius=10)
        sum_frame.pack(fill="x", pady=(6, 0), ipady=6)

        ctk.CTkLabel(sum_frame, text="Session stats:",
                     text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 10)).pack(side="left", padx=10)
        self._stats_var = tk.StringVar(value="Rallies: 0  |  IN: 0  |  OUT: 0")
        ctk.CTkLabel(sum_frame, textvariable=self._stats_var,
                     text_color=TEXT, fg_color="transparent",
                     font=("Consolas", 10)).pack(side="left")

    # ── Widget helper ─────────────────────────────────────────────

    def _btn(self, parent, text, cmd, accent):
        return ctk.CTkButton(parent, text=text, command=cmd,
                              text_color=accent, fg_color=BG3, hover_color=BG4,
                              corner_radius=8,
                              font=(FONT_FAMILY, 11, "bold"),
                              height=36)

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
        cm_str    = f"{cm[0]:+.0f}px" if cm else ""
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
        win = ctk.CTkToplevel(self._root)
        win.title("Edit Names")
        win.configure(fg_color=BG)
        win.geometry("360x200")
        win.resizable(False, False)
        win.transient(self._root)

        ctk.CTkLabel(win, text="Player A name:", text_color=TEXT, fg_color="transparent",
                     font=(FONT_FAMILY, 12)).grid(row=0, column=0, padx=14, pady=12, sticky="w")
        ea = ctk.CTkEntry(win, font=(FONT_FAMILY, 12), fg_color=BG3, text_color=WHITE,
                           width=170, corner_radius=8)
        ea.insert(0, self.name_a)
        ea.grid(row=0, column=1, padx=10, pady=12)

        ctk.CTkLabel(win, text="Player B name:", text_color=TEXT, fg_color="transparent",
                     font=(FONT_FAMILY, 12)).grid(row=1, column=0, padx=14, pady=6, sticky="w")
        eb = ctk.CTkEntry(win, font=(FONT_FAMILY, 12), fg_color=BG3, text_color=WHITE,
                           width=170, corner_radius=8)
        eb.insert(0, self.name_b)
        eb.grid(row=1, column=1, padx=10, pady=6)

        def _apply():
            self.name_a = ea.get().strip() or "Player A"
            self.name_b = eb.get().strip() or "Player B"
            self._name_a_var.set(self.name_a)
            self._name_b_var.set(self.name_b)
            self._refresh_ui()
            win.destroy()

        ctk.CTkButton(win, text="Save", command=_apply,
                      text_color=WHITE, fg_color=CYAN, hover_color="#3d8bd6",
                      corner_radius=8, font=(FONT_FAMILY, 12, "bold")).grid(
                      row=2, column=0, columnspan=2, pady=16)

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
            f"Umpire: {self.umpire_name} ({self.role})",
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
        self._score_a_label.configure(
            text_color=WHITE if self.serve=="A" else SCORE_A_COL)
        self._score_b_label.configure(
            text_color=WHITE if self.serve=="B" else SCORE_B_COL)

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
        self._decision_lbl.configure(text_color=color)
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
