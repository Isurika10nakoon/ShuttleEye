# score_system.py
import cv2


class ScoreSystem:
    """
    Rally-point scoring with game/set tracking.
    IN  → Player A scores  (receiving side wins the rally)
    OUT → Player B scores  (serving side wins the rally)
    """

    WINNING_SCORE = 21
    DEUCE_SCORE   = 20
    MAX_SCORE     = 30

    def __init__(self):
        self.playerA    = 0
        self.playerB    = 0
        self.game_over  = False
        self.winner     = ""
        self.last_point = ""   # "A" or "B"

    def update(self, decision):
        if self.game_over or not decision:
            return
        if decision == "IN":
            self.playerA   += 1
            self.last_point = "A"
        elif decision == "OUT":
            self.playerB   += 1
            self.last_point = "B"
        self._check_winner()

    def _check_winner(self):
        a, b = self.playerA, self.playerB
        # Standard: first to 21 with 2-point lead; cap at 30
        if a >= self.WINNING_SCORE or b >= self.WINNING_SCORE:
            if abs(a - b) >= 2 or max(a, b) >= self.MAX_SCORE:
                self.game_over = True
                self.winner    = "Player A" if a > b else "Player B"

    def reset(self):
        self.playerA    = 0
        self.playerB    = 0
        self.game_over  = False
        self.winner     = ""
        self.last_point = ""

    def get_score(self):
        return self.playerA, self.playerB

    # ── Drawing ──────────────────────────────────────────────────

    def draw_score(self, frame):
        h, w = frame.shape[:2]
        a, b = self.get_score()

        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-320, 0), (w, 130), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Highlight last scorer
        col_a = (0, 255, 120) if self.last_point == "A" else (180, 180, 180)
        col_b = (80, 120, 255) if self.last_point == "B" else (180, 180, 180)

        cv2.putText(frame, f"Player A: {a}", (w-308, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_a, 2)
        cv2.putText(frame, f"Player B: {b}", (w-308, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col_b, 2)

        # Deuce label
        if a >= self.DEUCE_SCORE and b >= self.DEUCE_SCORE:
            cv2.putText(frame, "DEUCE", (w-308, 118),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

        # Game-over banner
        if self.game_over:
            banner = f"{self.winner} WINS!"
            bw     = cv2.getTextSize(banner, cv2.FONT_HERSHEY_DUPLEX, 2.2, 6)[0][0]
            bx     = (w - bw) // 2
            cv2.putText(frame, banner, (bx+3, h//2+3),
                        cv2.FONT_HERSHEY_DUPLEX, 2.2, (0,0,0), 9)
            cv2.putText(frame, banner, (bx, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2.2, (0,215,255), 6)

        return frame
