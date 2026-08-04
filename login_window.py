# login_window.py — ShuttleEye login + admin account management
# ═══════════════════════════════════════════════════════════════════════
#  show_login()       → blocks, returns (username, role) or None
#  show_admin_panel()  → blocks, returns "continue" or "logout"
# ═══════════════════════════════════════════════════════════════════════

import customtkinter as ctk
from tkinter import messagebox

import auth

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG       = "#0d1117"
CARD     = "#161b22"
CARD2    = "#21262d"
BORDER   = "#30363d"
TEXT     = "#e6edf3"
TEXT_DIM = "#8b949e"
CYAN     = "#58a6ff"
GREEN    = "#3fb950"
RED      = "#f85149"
YELLOW   = "#d29922"

FONT_FAMILY = "Segoe UI"


class LoginWindow(ctk.CTk):
    """Login screen. Blocks until a user logs in or closes the window."""

    def __init__(self):
        super().__init__()
        self.title("ShuttleEye — Login")
        self.geometry("440x560")
        self.resizable(False, False)
        self.configure(fg_color=BG)
        self.result = None  # (username, role) on success

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=20,
                             border_width=1, border_color=BORDER)
        card.pack(expand=True, fill="both", padx=32, pady=40)

        ctk.CTkLabel(card, text="🏸", font=(FONT_FAMILY, 48),
                     fg_color="transparent").pack(pady=(40, 6))
        ctk.CTkLabel(card, text="ShuttleEye", font=(FONT_FAMILY, 28, "bold"),
                     text_color=CYAN, fg_color="transparent").pack()
        ctk.CTkLabel(card, text="Umpire & Line-Judge System",
                     font=(FONT_FAMILY, 12), text_color=TEXT_DIM,
                     fg_color="transparent").pack(pady=(0, 32))

        self._user_entry = ctk.CTkEntry(card, placeholder_text="Username",
                                         width=270, height=44, corner_radius=10,
                                         fg_color=CARD2, border_color=BORDER,
                                         font=(FONT_FAMILY, 13))
        self._user_entry.pack(pady=6)

        self._pass_entry = ctk.CTkEntry(card, placeholder_text="Password", show="•",
                                         width=270, height=44, corner_radius=10,
                                         fg_color=CARD2, border_color=BORDER,
                                         font=(FONT_FAMILY, 13))
        self._pass_entry.pack(pady=6)
        self._pass_entry.bind("<Return>", lambda e: self._attempt_login())
        self._user_entry.bind("<Return>", lambda e: self._pass_entry.focus_set())

        self._error_var = ctk.StringVar(value="")
        ctk.CTkLabel(card, textvariable=self._error_var, text_color=RED,
                     font=(FONT_FAMILY, 11), fg_color="transparent").pack(pady=(8, 0))

        ctk.CTkButton(card, text="Log In", command=self._attempt_login,
                      width=270, height=44, corner_radius=10,
                      fg_color=CYAN, hover_color="#3d8bd6",
                      font=(FONT_FAMILY, 14, "bold")).pack(pady=(18, 6))

        ctk.CTkLabel(card, text="Default accounts:  admin / admin123   ·   umpire / umpire123",
                     font=(FONT_FAMILY, 9), text_color=TEXT_DIM,
                     fg_color="transparent").pack(pady=(28, 0))

        self._user_entry.focus_set()

    def _attempt_login(self):
        username = self._user_entry.get().strip()
        password = self._pass_entry.get()
        role = auth.authenticate(username, password)
        if role is None:
            self._error_var.set("Invalid username or password.")
            self._pass_entry.delete(0, "end")
            return
        self.result = (username, role)
        self.destroy()

    def _on_close(self):
        self.result = None
        self.destroy()


def show_login():
    """Blocks until login succeeds or the window is closed.
    Returns (username, role) or None."""
    win = LoginWindow()
    win.mainloop()
    return win.result


# ═══════════════════════════════════════════════════════════════════════
#  Admin panel — manage umpire accounts
# ═══════════════════════════════════════════════════════════════════════

class AdminPanel(ctk.CTk):
    def __init__(self, admin_username):
        super().__init__()
        self.admin_username = admin_username
        self.result = "logout"  # default if window is closed

        self.title("ShuttleEye — Admin Panel")
        self.geometry("520x620")
        self.minsize(480, 520)
        self.configure(fg_color=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(18, 6))
        ctk.CTkLabel(header, text=f"👋 Welcome, {self.admin_username}",
                     font=(FONT_FAMILY, 18, "bold"), text_color=CYAN,
                     fg_color="transparent").pack(anchor="w")
        ctk.CTkLabel(header, text="Manage umpire accounts",
                     font=(FONT_FAMILY, 11), text_color=TEXT_DIM,
                     fg_color="transparent").pack(anchor="w")

        # Umpire list
        list_card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=14,
                                  border_width=1, border_color=BORDER)
        list_card.pack(fill="both", expand=True, padx=20, pady=10)

        ctk.CTkLabel(list_card, text="UMPIRES", text_color=TEXT_DIM,
                     fg_color="transparent",
                     font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", padx=14, pady=(12, 4))

        self._scroll = ctk.CTkScrollableFrame(list_card, fg_color="transparent")
        self._scroll.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Add umpire form
        add_card = ctk.CTkFrame(self, fg_color=CARD, corner_radius=14,
                                 border_width=1, border_color=BORDER)
        add_card.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(add_card, text="ADD NEW UMPIRE", text_color=TEXT_DIM,
                     fg_color="transparent",
                     font=(FONT_FAMILY, 10, "bold")).grid(row=0, column=0, columnspan=3,
                                                           sticky="w", padx=14, pady=(12, 6))

        self._new_user_entry = ctk.CTkEntry(add_card, placeholder_text="Username",
                                             fg_color=CARD2, corner_radius=8, width=150)
        self._new_user_entry.grid(row=1, column=0, padx=(14, 6), pady=(0, 14))

        self._new_pass_entry = ctk.CTkEntry(add_card, placeholder_text="Password", show="•",
                                             fg_color=CARD2, corner_radius=8, width=150)
        self._new_pass_entry.grid(row=1, column=1, padx=6, pady=(0, 14))

        ctk.CTkButton(add_card, text="＋ Add", command=self._add_umpire,
                      fg_color=GREEN, hover_color="#2ea043", corner_radius=8,
                      width=80, font=(FONT_FAMILY, 12, "bold")).grid(
                      row=1, column=2, padx=(6, 14), pady=(0, 14))
        add_card.columnconfigure(0, weight=1)
        add_card.columnconfigure(1, weight=1)

        # Bottom actions
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", padx=20, pady=(0, 20))

        ctk.CTkButton(bottom, text="Logout", command=self._logout,
                      fg_color=CARD2, hover_color=BORDER, text_color=TEXT_DIM,
                      corner_radius=10, height=42, font=(FONT_FAMILY, 12, "bold")
                      ).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ctk.CTkButton(bottom, text="Continue to Dashboard →", command=self._continue,
                      fg_color=CYAN, hover_color="#3d8bd6",
                      corner_radius=10, height=42, font=(FONT_FAMILY, 13, "bold")
                      ).pack(side="left", expand=True, fill="x", padx=(6, 0))

    def _refresh_list(self):
        for w in self._scroll.winfo_children():
            w.destroy()

        umpires = auth.list_users(role="umpire")
        if not umpires:
            ctk.CTkLabel(self._scroll, text="No umpire accounts yet.",
                         text_color=TEXT_DIM, fg_color="transparent",
                         font=(FONT_FAMILY, 11)).pack(pady=10)
            return

        for username in sorted(umpires):
            row = ctk.CTkFrame(self._scroll, fg_color=CARD2, corner_radius=10)
            row.pack(fill="x", pady=4)

            ctk.CTkLabel(row, text=f"👤  {username}", text_color=TEXT,
                         fg_color="transparent",
                         font=(FONT_FAMILY, 12, "bold")).pack(side="left", padx=12, pady=10)

            ctk.CTkButton(row, text="Remove", command=lambda u=username: self._remove_umpire(u),
                          fg_color="transparent", hover_color=BORDER, text_color=RED,
                          corner_radius=8, width=70, height=28,
                          font=(FONT_FAMILY, 10)).pack(side="right", padx=(4, 10), pady=6)
            ctk.CTkButton(row, text="Reset PW", command=lambda u=username: self._reset_pw(u),
                          fg_color="transparent", hover_color=BORDER, text_color=YELLOW,
                          corner_radius=8, width=80, height=28,
                          font=(FONT_FAMILY, 10)).pack(side="right", padx=4, pady=6)

    def _add_umpire(self):
        username = self._new_user_entry.get().strip()
        password = self._new_pass_entry.get()
        try:
            auth.add_user(username, password, "umpire")
        except ValueError as e:
            messagebox.showerror("Cannot add umpire", str(e))
            return
        self._new_user_entry.delete(0, "end")
        self._new_pass_entry.delete(0, "end")
        self._refresh_list()

    def _remove_umpire(self, username):
        if not messagebox.askyesno("Remove umpire", f"Remove umpire account '{username}'?"):
            return
        auth.remove_user(username)
        self._refresh_list()

    def _reset_pw(self, username):
        dialog = ctk.CTkInputDialog(text=f"New password for '{username}':",
                                     title="Reset Password")
        new_password = dialog.get_input()
        if not new_password:
            return
        auth.reset_password(username, new_password)
        messagebox.showinfo("Password reset", f"Password updated for '{username}'.")

    def _logout(self):
        self.result = "logout"
        self.destroy()

    def _continue(self):
        self.result = "continue"
        self.destroy()

    def _on_close(self):
        self.result = "logout"
        self.destroy()


def show_admin_panel(admin_username):
    """Blocks. Returns 'continue' or 'logout'."""
    win = AdminPanel(admin_username)
    win.mainloop()
    return win.result


def run_login_flow():
    """
    Full login flow: repeats login → (admin hub) → until an umpire is ready
    to proceed to the dashboard, or the user gives up.
    Returns (username, role) to launch the dashboard with, or None to quit.
    """
    while True:
        login_result = show_login()
        if login_result is None:
            return None

        username, role = login_result
        if role == "umpire":
            return username, role

        # Admin: show management hub, then decide whether to continue or re-login
        action = show_admin_panel(username)
        if action == "continue":
            return username, role
        # else: loop back to login screen
