# web_dashboard.py — ShuttleEye remote umpire dashboard
# ═══════════════════════════════════════════════════════════════════════
#  A small Flask server that mirrors and controls a running
#  UmpireDashboard over the network, so the umpire's own device (phone,
#  tablet, laptop — anything with a browser) doesn't need to be the same
#  machine that's running the camera + detection pipeline.
#
#  It talks to the UmpireDashboard purely through its already thread-safe
#  public API (add_point, subtract_point, toggle_serve, undo_last_point,
#  start_new_set, reset_match, set_names, set_match_format, get_state,
#  export_log_text) — no Tkinter objects are touched from this thread.
#
#  USAGE
#  ─────
#  from web_dashboard import run_web_dashboard
#  run_web_dashboard(dashboard)   # starts serving in a background thread
# ═══════════════════════════════════════════════════════════════════════

import secrets
import socket
import threading

from flask import Flask, jsonify, request, session, redirect, url_for, Response

import auth
import db

DEFAULT_PORT = 8080


def get_lan_ip():
    """Best-effort LAN IP so the umpire knows what address to browse to
    from their own device. Falls back to localhost if it can't tell."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def create_app(dashboard):
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)

    def logged_in():
        return "username" in session

    def is_admin():
        return session.get("role") == "admin"

    @app.get("/")
    def index():
        if not logged_in():
            return redirect(url_for("login"))
        # Admins land on the multi-court overview by default — scoring a
        # single match is still an umpire's job, done from their own
        # court's dashboard (desktop or this same page, logged in as
        # an umpire account).
        if is_admin():
            return redirect(url_for("admin_page"))
        return Response(PAGE_HTML, mimetype="text/html")

    @app.get("/login")
    def login():
        if logged_in():
            return redirect(url_for("index"))
        return Response(LOGIN_HTML, mimetype="text/html")

    @app.get("/admin")
    def admin_page():
        if not logged_in():
            return redirect(url_for("login"))
        if not is_admin():
            return redirect(url_for("index"))
        return Response(ADMIN_HTML, mimetype="text/html")

    @app.get("/board")
    def board_page():
        # Intentionally public — meant to be left open on a courtside
        # TV/projector for spectators, no umpire/admin account needed.
        return Response(BOARD_HTML, mimetype="text/html")

    @app.get("/api/board")
    def api_board():
        # Public, read-only subset of the match state — no controls, no
        # login. Safe to expose: it's the same score anyone courtside
        # can already see by looking at the players.
        state = dashboard.get_state()
        return jsonify({
            "court_name"  : state["court_name"],
            "name_a"      : state["name_a"],
            "name_b"      : state["name_b"],
            "score_a"     : state["score_a"],
            "score_b"     : state["score_b"],
            "sets_a"      : state["sets_a"],
            "sets_b"      : state["sets_b"],
            "set_history" : state["set_history"],
            "serve"       : state["serve"],
            "set_num"     : state["set_num"],
            "status"      : state["status"],
            "winning_score": state["winning_score"],
            "game_over"   : state["game_over"],
        })

    @app.post("/api/login")
    def api_login():
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""
        role = auth.authenticate(username, password)
        if role is None:
            return jsonify(ok=False, error="Invalid username or password."), 401
        session["username"] = username
        session["role"] = role
        return jsonify(ok=True)

    @app.post("/api/logout")
    def api_logout():
        session.clear()
        return jsonify(ok=True)

    def require_login():
        if not logged_in():
            return jsonify(ok=False, error="Not logged in."), 401
        return None

    def require_admin():
        if not logged_in():
            return jsonify(ok=False, error="Not logged in."), 401
        if not is_admin():
            return jsonify(ok=False, error="Admin only."), 403
        return None

    @app.get("/api/admin/matches")
    def api_admin_matches():
        unauthorized = require_admin()
        if unauthorized:
            return unauthorized
        active = [dict(r) for r in db.list_active_matches()]
        recent = [dict(r) for r in db.list_recent_matches()]
        for row in active + recent:
            for key in ("started_at", "ended_at"):
                if row.get(key) is not None:
                    row[key] = row[key].isoformat()
        return jsonify(active=active, recent=recent)

    @app.get("/api/state")
    def api_state():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        state = dashboard.get_state()
        state["viewer"] = session["username"]
        return jsonify(state)

    @app.post("/api/point")
    def api_point():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        data = request.get_json(silent=True) or {}
        side = data.get("side")
        delta = data.get("delta")
        if side not in ("A", "B") or delta not in (1, -1):
            return jsonify(ok=False, error="Invalid side/delta."), 400
        if delta == 1:
            dashboard.add_point(side)
        else:
            dashboard.subtract_point(side)
        return jsonify(ok=True)

    @app.post("/api/toggle_serve")
    def api_toggle_serve():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        dashboard.toggle_serve()
        return jsonify(ok=True)

    @app.post("/api/undo")
    def api_undo():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        dashboard.undo_last_point()
        return jsonify(ok=True)

    @app.post("/api/new_set")
    def api_new_set():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        dashboard.start_new_set(force=True)
        return jsonify(ok=True)

    @app.post("/api/reset_match")
    def api_reset_match():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        dashboard.reset_match(force=True)
        return jsonify(ok=True)

    @app.post("/api/names")
    def api_names():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        data = request.get_json(silent=True) or {}
        dashboard.set_names(data.get("name_a", ""), data.get("name_b", ""))
        return jsonify(ok=True)

    @app.post("/api/match_format")
    def api_match_format():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        data = request.get_json(silent=True) or {}
        points = data.get("points")
        if points not in (21, 15):
            return jsonify(ok=False, error="points must be 21 or 15."), 400
        dashboard.set_match_format(points)
        return jsonify(ok=True)

    @app.get("/api/export")
    def api_export():
        unauthorized = require_login()
        if unauthorized:
            return unauthorized
        text = dashboard.export_log_text()
        return Response(
            text, mimetype="text/plain",
            headers={"Content-Disposition": "attachment; filename=shuttleeye_rally_log.txt"},
        )

    return app


def run_web_dashboard(dashboard, host="0.0.0.0", port=DEFAULT_PORT):
    """Start serving the remote dashboard in a background thread.
    Returns the LAN URL the umpire's device should open."""
    app = create_app(dashboard)
    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True),
        daemon=True,
    )
    thread.start()
    return f"http://{get_lan_ip()}:{port}"


# ═══════════════════════════════════════════════════════════════════════
#  Front-end — a single self-contained mobile-friendly page per view,
#  styled to match the desktop dashboard's dark theme.
# ═══════════════════════════════════════════════════════════════════════

_STYLE = """
:root{color-scheme:dark;}
*{box-sizing:border-box;}
body{margin:0;background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;
     -webkit-tap-highlight-color:transparent;}
.wrap{max-width:480px;margin:0 auto;padding:16px 16px 40px;min-height:100vh;}
.card{background:#161b22;border:1px solid #30363d;border-radius:16px;padding:16px;margin-bottom:12px;}
h1{font-size:1.2rem;color:#58a6ff;margin:0;}
.sub{color:#8b949e;font-size:.8rem;margin-top:2px;}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:14px 4px;}
.badge{background:#21262d;border-radius:8px;padding:4px 10px;font-size:.75rem;font-weight:bold;}
.badge.admin{color:#d29922;} .badge.umpire{color:#3fb950;}
.score-row{display:flex;align-items:center;justify-content:space-between;text-align:center;}
.score-col{flex:1;}
.name{font-weight:bold;font-size:1rem;}
.name.a{color:#58a6ff;} .name.b{color:#f0883e;}
.score{font-size:4rem;font-weight:bold;line-height:1;margin:6px 0;}
.score.a{color:#58a6ff;} .score.b{color:#f0883e;}
.score.serving{color:#fff;}
.vs{color:#8b949e;font-size:1.4rem;padding:0 8px;}
.status{text-align:center;color:#d29922;font-weight:bold;min-height:1.4em;margin-top:4px;}
.serve-line{text-align:center;color:#d29922;font-size:.85rem;margin-top:4px;}
.meta{display:flex;justify-content:space-between;font-size:.8rem;color:#8b949e;padding:6px 4px;}
.btn-row{display:flex;gap:8px;margin-bottom:8px;}
button{flex:1;border:none;border-radius:12px;padding:16px 8px;font-size:1.05rem;font-weight:bold;
       color:#fff;cursor:pointer;font-family:inherit;}
button:active{filter:brightness(0.85);}
.pt-a{background:#58a6ff;} .pt-b{background:#f0883e;}
.sub-a{background:transparent;border:2px solid #58a6ff;color:#58a6ff;}
.sub-b{background:transparent;border:2px solid #f0883e;color:#f0883e;}
.util{background:#21262d;color:#e6edf3;padding:12px 8px;font-size:.85rem;}
.util.serve{color:#8b949e;} .util.undo{color:#f0883e;}
.util.new-set{background:#58a6ff;} .util.reset{background:#f85149;}
.small{background:#21262d;color:#8b949e;font-size:.75rem;padding:8px;}
.names-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.edit-btn{flex:0 0 auto;padding:6px 14px;font-size:.8rem;background:#21262d;color:#58a6ff;
          border-radius:8px;}
.edit-form{display:none;margin-bottom:10px;}
.edit-form.open{display:block;}
input{width:100%;padding:12px;border-radius:10px;border:1px solid #30363d;background:#21262d;
      color:#fff;font-size:1rem;margin-bottom:10px;font-family:inherit;}
label{font-size:.75rem;color:#8b949e;}
.err{color:#f85149;font-size:.85rem;min-height:1.2em;text-align:center;margin-top:8px;}
.login-card{margin-top:15vh;text-align:center;}
.emoji{font-size:2.6rem;}

/* Admin — multi-court overview */
.wrap.admin-wrap{max-width:900px;}
#courts{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;margin-bottom:16px;}
.court-card .score{font-size:2.3rem;}
.court-card .names-row{margin-bottom:2px;}
.recent-row{padding:8px 0;border-bottom:1px solid #30363d;font-size:.85rem;}
.recent-row:last-child{border-bottom:none;}
.recent-row .win{color:#3fb950;} .recent-row .aband{color:#8b949e;}
.empty{color:#8b949e;font-size:.9rem;text-align:center;padding:12px;}
"""

LOGIN_HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>ShuttleEye — Login</title><style>{_STYLE}</style></head>
<body><div class="wrap"><div class="card login-card">
<div class="emoji">🏸</div>
<h1>ShuttleEye</h1>
<div class="sub">Remote Umpire Dashboard</div>
<div style="height:20px"></div>
<input id="username" placeholder="Username" autocomplete="username">
<input id="password" type="password" placeholder="Password" autocomplete="current-password">
<button class="pt-a" style="width:100%" onclick="doLogin()">Log In</button>
<div class="err" id="err"></div>
</div></div>
<script>
async function doLogin(){{
  const username = document.getElementById('username').value;
  const password = document.getElementById('password').value;
  const res = await fetch('/api/login', {{method:'POST', headers:{{'Content-Type':'application/json'}},
                                           body: JSON.stringify({{username, password}})}});
  const data = await res.json();
  if (data.ok) {{ location.href = '/'; }}
  else {{ document.getElementById('err').textContent = data.error || 'Login failed'; }}
}}
document.getElementById('password').addEventListener('keydown', e => {{ if(e.key==='Enter') doLogin(); }});
</script>
</body></html>"""

PAGE_HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>ShuttleEye — Umpire Dashboard</title><style>{_STYLE}</style></head>
<body><div class="wrap">
<div class="topbar">
  <div><h1>🏸 ShuttleEye</h1><div class="sub" id="formatLabel">Race to 21</div></div>
  <div class="badge" id="roleBadge">…</div>
</div>

<div class="card">
  <div class="names-row">
    <span class="sub">Player / Team names</span>
    <button class="edit-btn" onclick="toggleEditNames()">✏ Edit</button>
  </div>
  <div class="edit-form" id="editNamesForm">
    <input id="editNameA" placeholder="Player A name">
    <input id="editNameB" placeholder="Player B name">
    <div class="btn-row">
      <button class="pt-a" onclick="saveNames()">Save</button>
      <button class="util" onclick="toggleEditNames(false)">Cancel</button>
    </div>
  </div>
  <div class="score-row">
    <div class="score-col"><div class="name a" id="nameA">Player A</div>
      <div class="score a" id="scoreA">0</div></div>
    <div class="vs">–</div>
    <div class="score-col"><div class="name b" id="nameB">Player B</div>
      <div class="score b" id="scoreB">0</div></div>
  </div>
  <div class="serve-line" id="serveLine">🏸 Serving: —</div>
  <div class="status" id="status"></div>
</div>

<div class="card">
  <div class="btn-row">
    <button class="pt-a" onclick="point('A',1)">＋ POINT A</button>
    <button class="pt-b" onclick="point('B',1)">＋ POINT B</button>
  </div>
  <div class="btn-row">
    <button class="sub-a" onclick="point('A',-1)">－ POINT A</button>
    <button class="sub-b" onclick="point('B',-1)">－ POINT B</button>
  </div>
  <div class="btn-row">
    <button class="util serve" onclick="action('toggle_serve')">🔄 Toggle Serve</button>
    <button class="util undo" onclick="action('undo')">↩ Undo Last Pt</button>
  </div>
  <div class="btn-row">
    <button class="util new-set" onclick="confirmAction('new_set','Start a new set now?')">🔁 New Set</button>
    <button class="util reset" onclick="confirmAction('reset_match','Reset the whole match?')">🗑 Reset Match</button>
  </div>
</div>

<div class="card small">
  <div id="setHistory">Set 1 in progress</div>
  <div id="setsWon" style="margin-top:4px"></div>
</div>

<div class="btn-row">
  <button class="util" style="flex:1" onclick="window.location='/api/export'">💾 Export Log</button>
  <button class="util" style="flex:1" onclick="window.open('/board','_blank')">📺 Spectator Board</button>
</div>
<div class="btn-row">
  <button class="util" style="flex:1" onclick="logout()">🚪 Log Out</button>
</div>
</div>

<script>
let lastState = null;

async function refresh(){{
  const res = await fetch('/api/state');
  if (res.status === 401) {{ location.href = '/login'; return; }}
  const s = await res.json();
  lastState = s;
  document.getElementById('nameA').textContent = s.name_a;
  document.getElementById('nameB').textContent = s.name_b;
  document.getElementById('scoreA').textContent = s.score_a;
  document.getElementById('scoreB').textContent = s.score_b;
  document.getElementById('scoreA').className = 'score a' + (s.serve==='A' ? ' serving' : '');
  document.getElementById('scoreB').className = 'score b' + (s.serve==='B' ? ' serving' : '');
  const servingName = s.serve === 'A' ? s.name_a : s.name_b;
  document.getElementById('serveLine').textContent = '🏸 Serving: ' + servingName;
  document.getElementById('status').textContent = s.status || '';
  document.getElementById('formatLabel').textContent = '🎯 Race to ' + s.winning_score;
  const badge = document.getElementById('roleBadge');
  badge.textContent = s.role.toUpperCase();
  badge.className = 'badge ' + s.role;
  document.getElementById('setsWon').textContent =
    s.name_a + ': ' + s.sets_a + '    ' + s.name_b + ': ' + s.sets_b;
  document.getElementById('setHistory').textContent = s.set_history.length
    ? s.set_history.map((p,i) => 'Set'+(i+1)+': '+p[0]+'–'+p[1]).join('   ')
    : 'Set ' + s.set_num + ' in progress';
}}

async function point(side, delta){{
  await fetch('/api/point', {{method:'POST', headers:{{'Content-Type':'application/json'}},
                              body: JSON.stringify({{side, delta}})}});
  refresh();
}}
async function action(name){{
  await fetch('/api/' + name, {{method:'POST'}});
  refresh();
}}
function confirmAction(name, msg){{
  if (confirm(msg)) action(name);
}}

function toggleEditNames(show){{
  const form = document.getElementById('editNamesForm');
  const willShow = show !== undefined ? show : !form.classList.contains('open');
  if (willShow && lastState) {{
    document.getElementById('editNameA').value = lastState.name_a;
    document.getElementById('editNameB').value = lastState.name_b;
  }}
  form.classList.toggle('open', willShow);
}}
async function saveNames(){{
  const name_a = document.getElementById('editNameA').value;
  const name_b = document.getElementById('editNameB').value;
  await fetch('/api/names', {{method:'POST', headers:{{'Content-Type':'application/json'}},
                              body: JSON.stringify({{name_a, name_b}})}});
  toggleEditNames(false);
  refresh();
}}

async function logout(){{
  await fetch('/api/logout', {{method:'POST'}});
  location.href = '/login';
}}

refresh();
setInterval(refresh, 1000);
</script>
</body></html>"""

ADMIN_HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>ShuttleEye — Admin</title><style>{_STYLE}</style></head>
<body><div class="wrap admin-wrap">
<div class="topbar">
  <div><h1>🏸 ShuttleEye</h1><div class="sub">All Courts — Live Overview</div></div>
  <div class="badge admin">ADMIN</div>
</div>

<div id="courts"><div class="empty">Loading…</div></div>

<div class="card" id="recentWrap" style="display:none">
  <div class="sub" style="margin-bottom:6px">RECENTLY FINISHED</div>
  <div id="recent"></div>
</div>

<div class="btn-row">
  <button class="util" style="flex:1" onclick="logout()">🚪 Log Out</button>
</div>
</div>

<script>
function escapeHtml(s){{
  return String(s).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
}}

async function refresh(){{
  const res = await fetch('/api/admin/matches');
  if (res.status === 401) {{ location.href = '/login'; return; }}
  if (res.status === 403) {{ location.href = '/'; return; }}
  const data = await res.json();

  const courtsEl = document.getElementById('courts');
  if (!data.active.length) {{
    courtsEl.innerHTML = '<div class="empty">No matches currently in progress.</div>';
  }} else {{
    courtsEl.innerHTML = data.active.map(m => `
      <div class="card court-card">
        <div class="names-row">
          <span class="sub">🏟 ${{escapeHtml(m.court_name)}}</span>
          <span class="sub">Race to ${{m.winning_score}}</span>
        </div>
        <div class="score-row">
          <div class="score-col"><div class="name a">${{escapeHtml(m.name_a)}}</div>
            <div class="score a">${{m.score_a}}</div></div>
          <div class="vs">–</div>
          <div class="score-col"><div class="name b">${{escapeHtml(m.name_b)}}</div>
            <div class="score b">${{m.score_b}}</div></div>
        </div>
        <div class="meta"><span>Set ${{m.set_num}}</span>
          <span>Sets ${{m.sets_a}}–${{m.sets_b}}</span>
          <span>👤 ${{escapeHtml(m.umpire_name)}}</span></div>
      </div>`).join('');
  }}

  const recentWrap = document.getElementById('recentWrap');
  if (data.recent.length) {{
    recentWrap.style.display = 'block';
    document.getElementById('recent').innerHTML = data.recent.map(m => {{
      const outcome = m.status === 'completed'
        ? `<span class="win">🏆 ${{escapeHtml(m.winner_name || '')}}</span>`
        : `<span class="aband">⚠ abandoned</span>`;
      return `<div class="recent-row">🏟 ${{escapeHtml(m.court_name)}} — `
           + `${{escapeHtml(m.name_a)}} ${{m.score_a}}–${{m.score_b}} ${{escapeHtml(m.name_b)}} `
           + `(sets ${{m.sets_a}}–${{m.sets_b}}) ${{outcome}}</div>`;
    }}).join('');
  }} else {{
    recentWrap.style.display = 'none';
  }}
}}

async function logout(){{
  await fetch('/api/logout', {{method:'POST'}});
  location.href = '/login';
}}

refresh();
setInterval(refresh, 2000);
</script>
</body></html>"""

BOARD_HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>ShuttleEye — Scoreboard</title>
<style>
:root{color-scheme:dark;}
*{box-sizing:border-box;}
html,body{height:100%;margin:0;background:#0d1117;color:#e6edf3;
          font-family:'Segoe UI',system-ui,sans-serif;overflow:hidden;
          -webkit-tap-highlight-color:transparent;}
.board{height:100vh;display:flex;flex-direction:column;align-items:center;
       justify-content:center;padding:2vh 3vw;text-align:center;cursor:none;}
.court{font-size:2.4vw;color:#8b949e;letter-spacing:.06em;margin-bottom:1vh;}
.format{font-size:1.6vw;color:#58a6ff;font-weight:bold;margin-bottom:3vh;}
.score-row{display:flex;align-items:center;justify-content:center;width:100%;gap:3vw;}
.col{flex:1;display:flex;flex-direction:column;align-items:center;min-width:0;}
.name{font-size:3.2vw;font-weight:bold;margin-bottom:1vh;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:95%;}
.name.a{color:#58a6ff;} .name.b{color:#f0883e;}
.score{font-size:16vw;font-weight:bold;line-height:1;}
.score.a{color:#58a6ff;} .score.b{color:#f0883e;}
.score.serving{color:#fff;text-shadow:0 0 40px currentColor;}
.dash{font-size:8vw;color:#30363d;padding:0 1vw;}
.serve{font-size:1.8vw;color:#d29922;margin-top:3vh;min-height:1.4em;}
.status{font-size:2.2vw;color:#d29922;font-weight:bold;margin-top:1vh;min-height:1.3em;}
.sets{font-size:1.6vw;color:#8b949e;margin-top:2vh;}
.setHistory{font-size:1.1vw;color:#8b949e;margin-top:.6vh;font-family:Consolas,monospace;}
.hint{position:fixed;bottom:10px;right:14px;font-size:.75rem;color:#30363d;}
.gameover{font-size:3vw;color:#3fb950;font-weight:bold;margin-top:2vh;}
</style></head>
<body>
<div class="board" id="board" onclick="goFullscreen()">
  <div class="court" id="court">🏸 SHUTTLEEYE</div>
  <div class="format" id="format">Race to 21</div>
  <div class="score-row">
    <div class="col"><div class="name a" id="nameA">Player A</div>
      <div class="score a" id="scoreA">0</div></div>
    <div class="dash">–</div>
    <div class="col"><div class="name b" id="nameB">Player B</div>
      <div class="score b" id="scoreB">0</div></div>
  </div>
  <div class="serve" id="serve"></div>
  <div class="status" id="status"></div>
  <div class="sets" id="sets"></div>
  <div class="setHistory" id="setHistory"></div>
</div>
<div class="hint">tap anywhere for fullscreen</div>
<script>
function goFullscreen(){
  if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(()=>{});
}

async function refresh(){
  try{
    const res = await fetch('/api/board');
    const s = await res.json();
    document.getElementById('court').textContent = '🏸 ' + s.court_name.toUpperCase();
    document.getElementById('format').textContent = 'Race to ' + s.winning_score;
    document.getElementById('nameA').textContent = s.name_a;
    document.getElementById('nameB').textContent = s.name_b;
    document.getElementById('scoreA').textContent = s.score_a;
    document.getElementById('scoreB').textContent = s.score_b;
    document.getElementById('scoreA').className = 'score a' + (s.serve==='A' ? ' serving' : '');
    document.getElementById('scoreB').className = 'score b' + (s.serve==='B' ? ' serving' : '');
    const servingName = s.serve === 'A' ? s.name_a : s.name_b;
    document.getElementById('serve').textContent = s.game_over ? '' : ('🏸 Serving: ' + servingName);
    document.getElementById('status').textContent = s.status || '';
    document.getElementById('sets').textContent =
      'Sets — ' + s.name_a + ': ' + s.sets_a + '    ' + s.name_b + ': ' + s.sets_b;
    document.getElementById('setHistory').textContent = s.set_history.length
      ? s.set_history.map((p,i) => 'Set'+(i+1)+': '+p[0]+'–'+p[1]).join('     ')
      : 'Set ' + s.set_num + ' in progress';
  } catch(e) { /* transient network hiccup — just retry next tick */ }
}

refresh();
setInterval(refresh, 1500);
</script>
</body></html>"""
