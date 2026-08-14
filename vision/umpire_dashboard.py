# umpire_dashboard.py  ─  ShuttleEye v5  ─  Umpire Scoring Dashboard
# ═══════════════════════════════════════════════════════════════════════
#
#  A professional CustomTkinter umpire dashboard that:
#  • Shows live score for both players / teams — the umpire can add or
#    subtract a point from either side at any time
#  • Tracks sets (games) with full set history
#  • Receives IN/OUT line calls from line_judge via callback — these are
#    informational only (badminton points aren't decided by a landing
#    call alone); the umpire always awards the point manually
#  • Keeps a rally log for export (hidden from the on-screen UI)
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
import webbrowser

import auth
import db
import bracket

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

    BEST_OF        = 3          # match is best of 3 sets — 3rd set only
                                 # played if each side has won one set

    def __init__(self, umpire_name="Umpire", role="umpire", court_name="Court 1"):
        self._q      = queue.Queue()   # thread-safe event queue
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._root   = None

        # Session / login info
        self.umpire_name = umpire_name
        self.role        = role
        self.court_name  = court_name

        # Set by app.py once the web dashboard is up, so the desktop
        # window can open the spectator board in a browser directly.
        self.web_url = None

        # Match format — points needed to win a set (21 or 15), chosen by
        # the umpire before the match starts. Deuce/cap scale with it.
        self.winning_score = 21
        self.deuce_score   = self.winning_score - 1
        self.max_score     = self.winning_score + 9

        # Match state
        self.score_a    = 0
        self.score_b    = 0
        self.sets_a     = 0
        self.sets_b     = 0
        self.set_history = []          # [(a_score, b_score), …]
        self.serve       = "A"         # "A" or "B"
        self.rally_log   = []          # list of log-entry dicts
        self.game_over   = False
        self.set_over    = False
        self.name_a      = "Player A"
        self.name_b      = "Player B"
        self._start_time = time.time()
        self._last_decision = None     # (decision, cm, px)
        self._set_num    = 1
        self.status_text = ""          # plain-Python mirror of _status_var,
                                        # safe to read from any thread (e.g.
                                        # the web dashboard)

        # Persistence — the DB row for the match currently in progress
        self._match_id = None

        # Set if this court has a tournament bracket match waiting to be
        # played — see _apply_match_format, which auto-fills the player
        # names from it, and _check_set_over, which reports the result
        # back to the bracket once the match finishes.
        self._bracket_slot = None

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

    def subtract_point(self, side):
        """Manually remove a point from side 'A' or 'B' (e.g. to correct
        a mis-click). Score is clamped at 0."""
        self._q.put(("POINT_MINUS", side))

    def toggle_serve(self):
        """Flip which side is currently serving."""
        self._q.put(("TOGGLE_SERVE", None))

    def undo_last_point(self):
        """Undo the most recent point change (an add or a subtract)."""
        self._q.put(("UNDO", None))

    def start_new_set(self, force=False):
        """Force-start a new set. With force=False (the default) this is
        ignored if the current set isn't finished — callers without a way
        to show a confirmation prompt (e.g. a remote web client) should
        confirm on their end first and pass force=True."""
        self._q.put(("NEW_SET", force))

    def reset_match(self, force=False):
        """Reset the whole match. With force=False (the default) this is
        ignored — callers without a way to show a confirmation prompt
        should confirm on their end first and pass force=True."""
        self._q.put(("RESET_MATCH", force))

    def set_names(self, name_a, name_b):
        """Set both player/team names directly."""
        self._q.put(("SET_NAMES", (name_a, name_b)))

    def set_match_format(self, points):
        """Set the match format (21 or 15 points) directly. Ignored once
        the match has already started."""
        self._q.put(("SET_FORMAT", points))

    def get_state(self):
        """Thread-safe snapshot of the match state as plain data — used by
        the web dashboard so an umpire's phone/tablet/laptop can mirror
        the same match a separate device is running. Safe to call from
        any thread; only reads plain Python attributes, never Tk widgets."""
        return {
            "umpire_name"    : self.umpire_name,
            "role"           : self.role,
            "court_name"     : self.court_name,
            "name_a"         : self.name_a,
            "name_b"         : self.name_b,
            "score_a"        : self.score_a,
            "score_b"        : self.score_b,
            "sets_a"         : self.sets_a,
            "sets_b"         : self.sets_b,
            "set_history"    : list(self.set_history),
            "serve"          : self.serve,
            "set_num"        : self._set_num,
            "game_over"      : self.game_over,
            "set_over"       : self.set_over,
            "status"         : self.status_text,
            "winning_score"  : self.winning_score,
            "elapsed_seconds": int(time.time() - self._start_time),
            "last_decision"  : self._last_decision[0] if self._last_decision else None,
        }

    def export_log_text(self):
        """Thread-safe: the rally log formatted as plain text, for the web
        dashboard's export/download action."""
        return "\n".join(self._export_lines())

    # ═══════════════════════════════════════════════════════════════
    #  Internal thread entry
    # ═══════════════════════════════════════════════════════════════

    def _run(self):
        self._root = ctk.CTk()
        self._root.title("ShuttleEye  ─  Umpire Dashboard")
        self._root.configure(fg_color=BG)
        self._size_to_screen()
        self._root.minsize(460, 600)  # still usable on small laptop screens
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._root.bind("<Configure>", self._on_resize)
        self._prompt_match_format()
        self._poll_queue()
        self._tick_timer()
        self._root.mainloop()

    def _size_to_screen(self):
        """Scale the initial window to the screen it's opening on and
        centre it, so the dashboard is usable on anything from a small
        laptop panel to a large desktop monitor."""
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        w  = max(460, min(720, int(sw * 0.40)))
        h  = max(600, min(920, int(sh * 0.85)))
        x  = (sw - w) // 2
        y  = (sh - h) // 3
        self._root.geometry(f"{w}x{h}+{x}+{y}")

    def _on_resize(self, event):
        if event.widget is not self._root:
            return
        if getattr(self, "_resize_after_id", None):
            self._root.after_cancel(self._resize_after_id)
        width = event.width
        self._resize_after_id = self._root.after(80, lambda: self._apply_responsive_fonts(width))

    def _apply_responsive_fonts(self, width):
        """Scale the big score digits with window width so they stay
        legible (and don't overflow) whether the window is small or huge."""
        size = max(40, min(96, width // 7))
        if getattr(self, "_score_font_size", None) == size:
            return
        self._score_font_size = size
        self._score_a_label.configure(font=(FONT_FAMILY, size, "bold"))
        self._score_b_label.configure(font=(FONT_FAMILY, size, "bold"))

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
                     text_color=CYAN, fg_color="transparent").pack(side="left", padx=(18, 8))

        ctk.CTkLabel(title_bar, text=f"🏟 {self.court_name}",
                     font=(FONT_FAMILY, 11, "bold"),
                     text_color=TEXT_DIM, fg_color="transparent").pack(side="left")

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

        self._format_var = tk.StringVar(value=f"🎯 Race to {self.winning_score}")
        ctk.CTkLabel(title_bar, textvariable=self._format_var,
                     font=(FONT_FAMILY, 11, "bold"),
                     text_color=CYAN, fg_color="transparent").pack(side="right", padx=8)

        tk.Frame(root, bg=BORDER, height=1).pack(fill="x")

        # ── Main body ────────────────────────────────────────────
        # The rally-log panel is intentionally not shown — it's still built
        # (see _build_right) so export/undo keep working, it's just never
        # packed into the visible window.
        body = ctk.CTkFrame(root, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Scrollable so every control stays reachable even if the window
        # ends up shorter than the content (small screens, high DPI, etc.)
        # — content used to silently clip below the visible window.
        left = ctk.CTkScrollableFrame(body, fg_color=BG, corner_radius=0)
        left.pack(side="left", fill="both", expand=True, padx=14, pady=10)

        hidden = ctk.CTkFrame(root, fg_color=BG, corner_radius=0, width=1, height=1)
        self._build_left(left)
        self._build_right(hidden)

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
        sb = ctk.CTkFrame(parent, fg_color=BG2, corner_radius=16,
                           border_width=1, border_color=BORDER)
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

        # Point buttons — the primary action, made large and bold so the
        # umpire can hit them quickly and reliably during play
        row1 = ctk.CTkFrame(ctrl, fg_color="transparent");  row1.pack(fill="x", pady=(3, 4))
        self._point_btn(row1, "＋ POINT  A", lambda: self.add_point("A"),
                  SCORE_A_COL, "#3d8bd6").pack(side="left", expand=True, fill="x", padx=3)
        self._point_btn(row1, "＋ POINT  B", lambda: self.add_point("B"),
                  SCORE_B_COL, "#d9722a").pack(side="left", expand=True, fill="x", padx=3)

        # Subtract-point buttons — corrects a mis-click without a full undo.
        # Same size as the add buttons above, just outlined instead of
        # filled so the two actions stay visually distinct.
        row2 = ctk.CTkFrame(ctrl, fg_color="transparent");  row2.pack(fill="x", pady=(0, 8))
        self._point_btn(row2, "－ POINT  A", lambda: self.subtract_point("A"),
                  SCORE_A_COL, BG4, filled=False).pack(side="left", expand=True, fill="x", padx=3)
        self._point_btn(row2, "－ POINT  B", lambda: self.subtract_point("B"),
                  SCORE_B_COL, BG4, filled=False).pack(side="left", expand=True, fill="x", padx=3)

        # Serve toggle / undo / reset — smaller than the point buttons,
        # just enough to stay clearly visible
        row4 = ctk.CTkFrame(ctrl, fg_color="transparent");  row4.pack(fill="x", pady=3)
        self._btn(row4, "🔄 Toggle Serve", self._toggle_serve,
                  TEXT_DIM, height=38, font_size=12).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row4, "↩ Undo Last Pt", self._undo_point,
                  ORANGE, height=38, font_size=12).pack(side="left", expand=True, fill="x", padx=2)

        # New Set / Reset Match — solid fill so they stand out clearly,
        # since these are higher-stakes actions the umpire needs to spot fast
        row5 = ctk.CTkFrame(ctrl, fg_color="transparent");  row5.pack(fill="x", pady=3)
        ctk.CTkButton(row5, text="🔁 New Set", command=self._new_set,
                      text_color=WHITE, fg_color=CYAN, hover_color="#3d8bd6",
                      corner_radius=10, font=(FONT_FAMILY, 12, "bold"),
                      height=38).pack(side="left", expand=True, fill="x", padx=2)
        ctk.CTkButton(row5, text="🗑 Reset Match", command=self._reset_match,
                      text_color=WHITE, fg_color=RED, hover_color="#c9463c",
                      corner_radius=10, font=(FONT_FAMILY, 12, "bold"),
                      height=38).pack(side="left", expand=True, fill="x", padx=2)

        row6 = ctk.CTkFrame(ctrl, fg_color="transparent");  row6.pack(fill="x", pady=3)
        self._btn(row6, "💾 Export Log", self._export_log,
                  TEXT_DIM, height=30, font_size=10).pack(side="left", expand=True, fill="x", padx=2)
        self._btn(row6, "⚙ Match Format", self._change_match_format,
                  TEXT_DIM, height=30, font_size=10).pack(side="left", expand=True, fill="x", padx=2)

        row7 = ctk.CTkFrame(ctrl, fg_color="transparent");  row7.pack(fill="x", pady=3)
        self._btn(row7, "📺 Open Spectator Board", self._open_spectator_board,
                  CYAN, height=30, font_size=10).pack(fill="x", padx=2)

    def _open_spectator_board(self):
        if not self.web_url:
            self._flash_decision("Web dashboard isn't running", RED)
            return
        webbrowser.open(f"{self.web_url}/board")

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

    def _btn(self, parent, text, cmd, accent, height=36, font_size=11):
        return ctk.CTkButton(parent, text=text, command=cmd,
                              text_color=accent, fg_color=BG3, hover_color=BG4,
                              corner_radius=8,
                              font=(FONT_FAMILY, font_size, "bold"),
                              height=height)

    def _point_btn(self, parent, text, cmd, color, hover, filled=True):
        if filled:
            return ctk.CTkButton(parent, text=text, command=cmd,
                                  text_color=WHITE, fg_color=color, hover_color=hover,
                                  corner_radius=14,
                                  font=(FONT_FAMILY, 18, "bold"),
                                  height=58)
        return ctk.CTkButton(parent, text=text, command=cmd,
                              text_color=color, fg_color="transparent", hover_color=hover,
                              corner_radius=14, border_width=2, border_color=color,
                              font=(FONT_FAMILY, 18, "bold"),
                              height=58)

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
                elif event == "POINT_MINUS":
                    self._subtract_point(data)
                elif event == "TOGGLE_SERVE":
                    self._toggle_serve()
                elif event == "UNDO":
                    self._undo_point()
                elif event == "NEW_SET":
                    self._new_set(force=data)
                elif event == "RESET_MATCH":
                    self._reset_match(force=data)
                elif event == "SET_NAMES":
                    name_a, name_b = data
                    self._set_names_internal(name_a, name_b)
                elif event == "SET_FORMAT":
                    if self._match_not_started():
                        self._apply_match_format(data)
                    else:
                        self._flash_decision("Reset the match to change format", RED)
        except queue.Empty:
            pass
        self._root.after(50, self._poll_queue)

    # ═══════════════════════════════════════════════════════════════
    #  Core logic
    # ═══════════════════════════════════════════════════════════════

    def _db_call(self, fn, *args, **kwargs):
        """Best-effort DB write — persistence must never take the live
        match down if PostgreSQL is briefly unreachable."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"[ShuttleEye] DB write failed ({fn.__name__}): {e}")
            return None

    def _handle_decision(self, decision, cm, px):
        """
        Receive a line-judge IN/OUT call. This is informational only —
        badminton points aren't decided by a landing call alone (faults,
        service errors, lets, etc. also apply) — so the score is never
        changed here. The umpire sees the call flashed on screen and
        decides whether/how to award the point via the Point A/B buttons.
        """
        self._last_decision = (decision, cm, px)
        color = GREEN if decision == "IN" else RED
        self._flash_decision(f"Line call: {decision}", color)

    def _apply_point(self, side, decision=None, cm=None, note=""):
        if self.game_over:
            return

        self._undo_stack_push()  # save the pre-change state

        if side == "A":
            self.score_a += 1
            self.serve    = "A"
        else:
            self.score_b += 1
            self.serve    = "B"

        self._log_rally(side, decision if decision else "manual", cm=cm, note=note)

        self._check_set_over()
        self._sync_live_score()
        self._refresh_ui()

    def _subtract_point(self, side):
        """Remove a point from `side` (e.g. to correct a mis-click).
        Ignored once the match is over or the side is already at 0."""
        if self.game_over:
            self._flash_decision("Match is over — reset to make changes", RED)
            return

        current = self.score_a if side == "A" else self.score_b
        if current <= 0:
            self._flash_decision(f"Score {side} already 0", RED)
            return

        self._undo_stack_push()  # save the pre-change state

        if side == "A":
            self.score_a -= 1
        else:
            self.score_b -= 1

        self._log_rally(side, "correction", note="point removed")
        self._sync_live_score()
        self._refresh_ui()

    def _sync_live_score(self):
        """Push the current in-set score to the DB so the admin's
        multi-court view stays near-live. Best-effort, non-blocking."""
        if self._match_id is not None:
            self._db_call(db.update_match_live_score,
                           self._match_id, self.score_a, self.score_b, self._set_num)

    def _log_rally(self, side, decision, cm=None, note=""):
        rally_num = len(self.rally_log) + 1
        ts        = datetime.datetime.now().strftime("%H:%M:%S")
        cm_str    = f"{cm[0]:+.0f}px" if cm else ""
        entry = {
            "num"     : rally_num,
            "ts"      : ts,
            "decision": decision,
            "side"    : side,
            "cm"      : cm_str,
            "score_a" : self.score_a,
            "score_b" : self.score_b,
            "note"    : note,
            "set"     : self._set_num,
        }
        self.rally_log.append(entry)
        self._add_log_row(entry)

        if self._match_id is not None:
            self._db_call(
                db.record_rally, self._match_id, self._set_num, rally_num,
                side, decision, cm_str, self.score_a, self.score_b, note,
            )

    def _check_set_over(self):
        a, b = self.score_a, self.score_b
        won  = False
        if a >= self.winning_score or b >= self.winning_score:
            if abs(a-b) >= 2 or max(a,b) >= self.max_score:
                won = True

        if not won:
            return

        winner = "A" if a > b else "B"
        if winner == "A":
            self.sets_a += 1
        else:
            self.sets_b += 1
        self.set_history.append((a, b))

        if self._match_id is not None:
            self._db_call(db.record_set, self._match_id, self._set_num, a, b, winner)
            self._db_call(db.update_match_progress, self._match_id, self.sets_a, self.sets_b)

        # Check match winner
        if self.sets_a > self.BEST_OF // 2 or self.sets_b > self.BEST_OF // 2:
            self.game_over   = True
            self.set_over    = True
            winner_side      = "A" if self.sets_a > self.sets_b else "B"
            match_winner     = self.name_a if winner_side == "A" else self.name_b
            self._set_status(f"🏆  {match_winner} WINS THE MATCH!")
            self._log_separator(f"MATCH WON BY {match_winner}")
            if self._match_id is not None:
                self._db_call(db.finish_match, self._match_id,
                               self.sets_a, self.sets_b, match_winner)
            if self._bracket_slot is not None:
                self._db_call(bracket.report_slot_result, self._bracket_slot, winner_side)
        else:
            # Set is over but the match continues — automatically advance
            # to the next set, resetting the score to 0-0.
            finished_set = self._set_num
            self._log_separator(f"SET {finished_set} END  {a}–{b}")
            self._flash_decision(f"Set {finished_set} complete — next set starting", CYAN)
            self._start_next_set()

    def _start_next_set(self):
        self._set_num  += 1
        self.score_a    = 0
        self.score_b    = 0
        self.set_over   = False
        self._set_status(f"Set {self._set_num} started")
        self._log_separator(f"─── SET {self._set_num} ───")
        self._undo_stack = []
        self._sync_live_score()

    def _new_set(self, force=False):
        """Force-start a new set (e.g. to abandon the current one early).
        Sets normally advance automatically once won. With force=False
        (local button clicks) an unfinished set asks for confirmation;
        force=True (e.g. a remote web client that already confirmed on
        its end) skips straight to it."""
        if not force and not self.set_over and not self.game_over:
            if not messagebox.askyesno("New Set", "Current set not finished. Start new set anyway?"):
                return
        self._start_next_set()
        self._refresh_ui()

    def _reset_match(self, force=False):
        if not force:
            if not messagebox.askyesno("Reset Match", "Reset all scores and rally log?"):
                return
        if self._match_id is not None and not self.game_over:
            self._db_call(db.abandon_match, self._match_id, self.sets_a, self.sets_b)
        self._match_id     = None
        self._bracket_slot = None
        self.score_a    = 0
        self.score_b    = 0
        self.sets_a     = 0
        self.sets_b     = 0
        self.set_history= []
        self.serve      = "A"
        self.game_over  = False
        self.set_over   = False
        self._set_num   = 1
        self.rally_log  = []
        self._start_time= time.time()
        self._set_status("")
        self._undo_stack= []
        self._log_list.delete(0, tk.END)
        self._refresh_ui()
        self._prompt_match_format()

    # ── Match format ─────────────────────────────────────────────

    def _match_not_started(self):
        return not (self.score_a or self.score_b or self.sets_a or self.sets_b
                    or self._set_num != 1)

    def _apply_match_format(self, points):
        self.winning_score = points
        self.deuce_score   = points - 1
        self.max_score     = points + 9
        self._format_var.set(f"🎯 Race to {points}")

        # If a tournament admin has assigned this court a bracket match,
        # pull the two names in automatically instead of Player A/B.
        self._bracket_slot = self._db_call(db.get_ready_slot_for_court, self.court_name)
        if self._bracket_slot is not None:
            self._set_names_internal(self._bracket_slot["name_a"], self._bracket_slot["name_b"])
            self._flash_decision(
                f"🏆 {self._bracket_slot['name_a']} vs {self._bracket_slot['name_b']}", CYAN)

        self._refresh_ui()

        umpire_id = self._db_call(auth.get_user_id, self.umpire_name)
        self._match_id = self._db_call(
            db.create_match, umpire_id, self.umpire_name, self.court_name,
            self.name_a, self.name_b, points,
        )
        self._sync_live_score()
        if self._bracket_slot is not None and self._match_id is not None:
            self._db_call(db.link_match_to_slot, self._bracket_slot["id"], self._match_id)

    def _change_match_format(self):
        """Umpire-triggered format change — only allowed before the match
        has actually started (no points/sets played yet)."""
        if not self._match_not_started():
            self._flash_decision("Reset the match to change format", RED)
            return
        self._prompt_match_format()

    def _prompt_match_format(self):
        win = ctk.CTkToplevel(self._root)
        win.title("Match Format")
        win.configure(fg_color=BG)
        win.geometry("380x220")
        win.resizable(False, False)
        win.transient(self._root)
        win.protocol("WM_DELETE_WINDOW", lambda: None)  # force a choice

        ctk.CTkLabel(win, text="🏸  Select match format", text_color=CYAN,
                     fg_color="transparent",
                     font=(FONT_FAMILY, 15, "bold")).pack(pady=(26, 4))
        ctk.CTkLabel(win, text="Points needed to win a set (best of 3 sets)",
                     text_color=TEXT_DIM, fg_color="transparent",
                     font=(FONT_FAMILY, 11)).pack(pady=(0, 20))

        def choose(points):
            self._apply_match_format(points)
            win.grab_release()
            win.destroy()

        row = ctk.CTkFrame(win, fg_color="transparent")
        row.pack()
        ctk.CTkButton(row, text="21 points", command=lambda: choose(21),
                      fg_color=CYAN, hover_color="#3d8bd6", corner_radius=10,
                      width=140, height=46,
                      font=(FONT_FAMILY, 14, "bold")).pack(side="left", padx=8)
        ctk.CTkButton(row, text="15 points", command=lambda: choose(15),
                      fg_color=BG3, hover_color=BG4, text_color=TEXT,
                      corner_radius=10, width=140, height=46,
                      font=(FONT_FAMILY, 14, "bold")).pack(side="left", padx=8)

        win.grab_set()
        self._root.wait_window(win)

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
        self._set_status("")
        self._sync_live_score()
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
            self._set_names_internal(ea.get(), eb.get())
            win.destroy()

        ctk.CTkButton(win, text="Save", command=_apply,
                      text_color=WHITE, fg_color=CYAN, hover_color="#3d8bd6",
                      corner_radius=8, font=(FONT_FAMILY, 12, "bold")).grid(
                      row=2, column=0, columnspan=2, pady=16)

    def _set_names_internal(self, name_a, name_b):
        self.name_a = (name_a or "").strip() or "Player A"
        self.name_b = (name_b or "").strip() or "Player B"
        self._name_a_var.set(self.name_a)
        self._name_b_var.set(self.name_b)
        self._refresh_ui()
        if self._match_id is not None:
            self._db_call(db.update_match_names, self._match_id, self.name_a, self.name_b)

    # ── Export ────────────────────────────────────────────────────

    def _export_lines(self):
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
        return lines

    def _export_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"shuttleeye_rally_log_{datetime.date.today()}.txt"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self._export_lines()))
        self._flash_decision("Log saved", GREEN)

    # ═══════════════════════════════════════════════════════════════
    #  UI refresh
    # ═══════════════════════════════════════════════════════════════

    def _refresh_ui(self):
        a, b = self.score_a, self.score_b
        self._score_a_var.set(str(a))
        self._score_b_var.set(str(b))

        # Deuce / advantage status
        if a >= self.deuce_score and b >= self.deuce_score:
            if a == b:
                self._set_status("DEUCE")
            elif a > b:
                self._set_status(f"ADVANTAGE  {self.name_a}")
            else:
                self._set_status(f"ADVANTAGE  {self.name_b}")
        elif not self.set_over and not self.game_over:
            self._set_status("")

        # Score label flash colour
        self._score_a_label.configure(
            text_color=WHITE if self.serve=="A" else SCORE_A_COL)
        self._score_b_label.configure(
            text_color=WHITE if self.serve=="B" else SCORE_B_COL)

        # Serve
        sn = self.name_a if self.serve=="A" else self.name_b
        self._serve_var.set(f"🏸  Serving: {sn}")

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

    def _set_status(self, text):
        """Update the status label and its plain-Python mirror
        (self.status_text) that other threads/the web dashboard read."""
        self.status_text = text
        self._status_var.set(text)

    # ═══════════════════════════════════════════════════════════════
    #  Timer
    # ═══════════════════════════════════════════════════════════════

    def _tick_timer(self):
        elapsed = int(time.time() - self._start_time)
        m, s    = divmod(elapsed, 60)
        self._timer_var.set(f"{m:02d}:{s:02d}")
        self._root.after(1000, self._tick_timer)
