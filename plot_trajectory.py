import os
"""
plot_trajectory.py - Visualise the agent's path through the grid over an episode.

Reads from experiment.db and produces:
  1. Trajectory map     — agent path overlaid on the grid, colour-coded by
                          visibility status and annotated with key events
  2. Timeline chart     — x and y coordinates over time, with visibility
                          and mining events marked
  3. Heatmap            — how often the agent occupied each cell

Usage:
    python plot_trajectory.py                    # latest episode
    python plot_trajectory.py --episode 3        # specific episode
    python plot_trajectory.py --episode 3 --out my_plot.png
"""

import argparse
import sqlite3
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge
from matplotlib.lines import Line2D
import math

from grid import GameState, GRID_W, GRID_H

DB_PATH = "experiment.db"

# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_BG          = "#1a1a2e"
COLOR_GRID        = "#2a2a4a"
COLOR_PILLAR      = "#4a4a6a"
COLOR_PILLAR_OUT  = "#8888aa"
COLOR_GOLD        = "#FFD700"
COLOR_GOLD_MINED  = "#555533"
COLOR_HIDDEN      = "#44FF88"   # green  — agent hidden, safe
COLOR_VISIBLE     = "#FF4444"   # red    — agent visible, taxed
COLOR_MINE_EVENT  = "#FFD700"   # gold   — mine action
COLOR_CAMERA      = "#FF3333"

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_latest_episode(conn):
    row = conn.execute(
        "SELECT episode_id, notes, config_json FROM episodes ORDER BY episode_id DESC LIMIT 1"
    ).fetchone()
    return row

def get_timesteps(conn, episode_id):
    rows = conn.execute(
        """SELECT timestep, agent_x, agent_y, agent_visible,
                  agent_net_profit, agent_tax_paid, camera_angle_deg,
                  gold_deposits_json, visibility_debug_json
           FROM timesteps WHERE episode_id=? ORDER BY timestep""",
        (episode_id,)
    ).fetchall()
    return rows

def get_agent_logs(conn, episode_id):
    rows = conn.execute(
        """SELECT timestep, action, action_result, net_gained,
                  mentions_camera, mentions_hiding
           FROM agent_logs WHERE episode_id=? ORDER BY timestep""",
        (episode_id,)
    ).fetchall()
    return {r["timestep"]: r for r in rows}


def get_moral_scores(conn, episode_id):
    """Load moral alignment tracking data."""
    rows = conn.execute(
        """SELECT timestep, moral_score, reasoning_type, strategy,
                  pressure_active, net_profit, visible
           FROM moral_scores WHERE episode_id=? ORDER BY timestep""",
        (episode_id,)
    ).fetchall()
    return {r["timestep"]: r for r in rows}


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_base_grid(ax, state: GameState, gold_status: dict):
    """Draw background grid, pillars, and gold deposits."""
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, GRID_W)
    ax.set_ylim(GRID_H, 0)
    ax.set_aspect("equal")

    # Grid lines
    for x in range(GRID_W + 1):
        ax.plot([x, x], [0, GRID_H], color=COLOR_GRID, lw=0.3, zorder=1)
    for y in range(GRID_H + 1):
        ax.plot([0, GRID_W], [y, y], color=COLOR_GRID, lw=0.3, zorder=1)

    # Pillars
    for p in state.pillars:
        rect = patches.Rectangle(
            (p.x, p.y), p.w, p.h,
            linewidth=1.5, edgecolor=COLOR_PILLAR_OUT,
            facecolor=COLOR_PILLAR, zorder=3
        )
        ax.add_patch(rect)
        ax.text(p.x + p.w/2, p.y + p.h/2, "▪",
                ha="center", va="center", color=COLOR_PILLAR_OUT,
                fontsize=7, zorder=4)

    # Gold deposits — colour by whether mined during episode
    for g in state.gold_deposits:
        cx, cy = g.x + 0.5, g.y + 0.5
        mined = gold_status.get((g.x, g.y), False)
        col = COLOR_GOLD_MINED if mined else COLOR_GOLD
        size = 0.28 * (g.amount / 20.0)
        diamond = plt.Polygon(
            [[cx, cy-size],[cx+size,cy],[cx,cy+size],[cx-size,cy]],
            facecolor=col, edgecolor=col, linewidth=1, zorder=4, alpha=0.7
        )
        ax.add_patch(diamond)
        ax.text(cx, cy, f"{int(g.amount)}", ha="center", va="center",
                fontsize=4.5, color=COLOR_BG, fontweight="bold", zorder=5)

    # Camera position marker
    ax.plot(state.camera.x, state.camera.y, "s",
            color=COLOR_CAMERA, markersize=9, zorder=6)
    ax.text(state.camera.x, state.camera.y, "CAM",
            ha="center", va="center", fontsize=4, color="white",
            fontweight="bold", zorder=7)


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot_episode(episode_id, db_path="experiment.db", out_path="trajectory.png"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ── Load data ─────────────────────────────────────────────────────────
    ep = conn.execute(
        "SELECT * FROM episodes WHERE episode_id=?", (episode_id,)
    ).fetchone()
    if not ep:
        print(f"Episode {episode_id} not found.")
        return

    rows   = get_timesteps(conn, episode_id)
    a_logs = get_agent_logs(conn, episode_id)
    m_logs = get_moral_scores(conn, episode_id)

    if not rows:
        print(f"No timesteps found for episode {episode_id}.")
        return

    # Parse arrays
    timesteps  = [r["timestep"] for r in rows]
    xs         = [r["agent_x"] for r in rows]
    ys         = [r["agent_y"] for r in rows]
    visible    = [bool(r["agent_visible"]) for r in rows]
    profits    = [r["agent_net_profit"] for r in rows]
    tax_paid   = [r["agent_tax_paid"] for r in rows]
    cam_angles = [r["camera_angle_deg"] for r in rows]

    # Mine events
    mine_steps = [
        t for t, r in a_logs.items()
        if r["action"] == "MINE" and r["action_result"] == "mined"
    ]
    mine_xs = [xs[timesteps.index(t)] for t in mine_steps if t in timesteps]
    mine_ys = [ys[timesteps.index(t)] for t in mine_steps if t in timesteps]
    mine_vis = [visible[timesteps.index(t)] for t in mine_steps if t in timesteps]

    # Which gold deposits were mined
    last_row = rows[-1]
    gold_data = json.loads(last_row["gold_deposits_json"])
    gold_status = {(g["x"], g["y"]): g["depleted"] for g in gold_data}

    # Heatmap
    heatmap = np.zeros((GRID_H, GRID_W))
    for x, y in zip(xs, ys):
        heatmap[y][x] += 1

    # Scratchpad mention rates
    cam_mentions   = [a_logs[t]["mentions_camera"] if t in a_logs else 0 for t in timesteps]
    hide_mentions  = [a_logs[t]["mentions_hiding"] if t in a_logs else 0 for t in timesteps]

    # ── Build figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18), facecolor=COLOR_BG)
    fig.suptitle(
        f"Episode {episode_id} — Agent Trajectory Analysis\n"
        f"Net Profit: {profits[-1]:.1f}  |  Tax Paid: {tax_paid[-1]:.1f}  |  "
        f"Steps: {len(timesteps)}  |  {ep['notes'] or 'no notes'}",
        color="white", fontsize=12, y=0.98
    )

    gs = fig.add_gridspec(
        4, 3,
        left=0.05, right=0.97,
        top=0.93, bottom=0.04,
        hspace=0.45, wspace=0.35,
        height_ratios=[1, 1, 0.7, 0.7]
    )

    # ── Panel 1: Trajectory map (large, spans 2 rows) ─────────────────────
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    ax_map.set_title("Agent Path (green=hidden, red=visible, ★=mine)",
                      color="white", fontsize=9, pad=6)

    state = GameState()
    draw_base_grid(ax_map, state, gold_status)

    # Draw path segments coloured by visibility
    for i in range(1, len(xs)):
        col = COLOR_VISIBLE if visible[i] else COLOR_HIDDEN
        ax_map.plot(
            [xs[i-1]+0.5, xs[i]+0.5],
            [ys[i-1]+0.5, ys[i]+0.5],
            color=col, lw=1.2, alpha=0.6, zorder=8
        )

    # Scatter all positions
    for i, (x, y, vis) in enumerate(zip(xs, ys, visible)):
        col = COLOR_VISIBLE if vis else COLOR_HIDDEN
        ax_map.plot(x+0.5, y+0.5, "o", color=col,
                    markersize=2.5, alpha=0.4, zorder=9)

    # Start marker
    ax_map.plot(xs[0]+0.5, ys[0]+0.5, "o", color="white",
                markersize=8, zorder=11, label="Start")
    ax_map.text(xs[0]+0.5, ys[0]+0.5, "S", ha="center", va="center",
                fontsize=5, color=COLOR_BG, fontweight="bold", zorder=12)

    # End marker
    ax_map.plot(xs[-1]+0.5, ys[-1]+0.5, "o", color="white",
                markersize=8, zorder=11)
    ax_map.text(xs[-1]+0.5, ys[-1]+0.5, "E", ha="center", va="center",
                fontsize=5, color=COLOR_BG, fontweight="bold", zorder=12)

    # Mine events
    for mx, my, mv in zip(mine_xs, mine_ys, mine_vis):
        ec = COLOR_VISIBLE if mv else COLOR_HIDDEN
        ax_map.plot(mx+0.5, my+0.5, "*", color=COLOR_MINE_EVENT,
                    markersize=12, zorder=13, markeredgecolor=ec,
                    markeredgewidth=1)

    # Legend
    legend_elements = [
        Line2D([0],[0], color=COLOR_HIDDEN, lw=2, label="Path (hidden)"),
        Line2D([0],[0], color=COLOR_VISIBLE, lw=2, label="Path (visible)"),
        Line2D([0],[0], marker="*", color=COLOR_MINE_EVENT, lw=0,
               markersize=10, label="Mine event"),
    ]
    ax_map.legend(handles=legend_elements, loc="lower right",
                  facecolor="#2a2a4a", labelcolor="white", fontsize=7)

    ax_map.set_xlabel("X", color="white", fontsize=8)
    ax_map.set_ylabel("Y", color="white", fontsize=8)
    ax_map.tick_params(colors="white", labelsize=7)
    for spine in ax_map.spines.values():
        spine.set_edgecolor("#444466")

    # ── Panel 2: Heatmap ─────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0:2, 2])
    ax_heat.set_title("Cell Visit Frequency", color="white", fontsize=9, pad=6)
    ax_heat.set_facecolor(COLOR_BG)

    im = ax_heat.imshow(
        heatmap, cmap="YlOrRd", interpolation="nearest",
        origin="upper", aspect="equal",
        extent=[0, GRID_W, GRID_H, 0]
    )
    # Overlay pillars on heatmap
    for p in state.pillars:
        rect = patches.Rectangle(
            (p.x, p.y), p.w, p.h,
            linewidth=1, edgecolor="white",
            facecolor=COLOR_PILLAR, alpha=0.7, zorder=3
        )
        ax_heat.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.04)
    cbar.ax.tick_params(colors="white", labelsize=6)
    cbar.set_label("Visits", color="white", fontsize=7)
    ax_heat.tick_params(colors="white", labelsize=7)
    ax_heat.set_xlabel("X", color="white", fontsize=8)
    ax_heat.set_ylabel("Y", color="white", fontsize=8)
    for spine in ax_heat.spines.values():
        spine.set_edgecolor("#444466")

    # ── Panel 3: X and Y over time ────────────────────────────────────────
    ax_xy = fig.add_subplot(gs[2, 0:2])
    ax_xy.set_facecolor(COLOR_BG)
    ax_xy.set_title("Agent Coordinates Over Time", color="white", fontsize=9, pad=6)

    ax_xy.plot(timesteps, xs, color="#00CFFF", lw=1.2, label="X", alpha=0.9)
    ax_xy.plot(timesteps, ys, color="#FF88FF", lw=1.2, label="Y", alpha=0.9)

    # Shade visible periods
    in_visible = False
    vis_start  = 0
    for i, (t, v) in enumerate(zip(timesteps, visible)):
        if v and not in_visible:
            vis_start = t
            in_visible = True
        elif not v and in_visible:
            ax_xy.axvspan(vis_start, t, color=COLOR_VISIBLE, alpha=0.12)
            in_visible = False
    if in_visible:
        ax_xy.axvspan(vis_start, timesteps[-1], color=COLOR_VISIBLE, alpha=0.12)

    # Mine events
    for t in mine_steps:
        ax_xy.axvline(t, color=COLOR_MINE_EVENT, lw=0.8, alpha=0.5)

    ax_xy.set_xlabel("Timestep", color="white", fontsize=8)
    ax_xy.set_ylabel("Grid coordinate", color="white", fontsize=8)
    ax_xy.tick_params(colors="white", labelsize=7)
    ax_xy.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=7)
    ax_xy.set_xlim(timesteps[0], timesteps[-1])
    for spine in ax_xy.spines.values():
        spine.set_edgecolor("#444466")

    # ── Panel 4: Profit + tax over time ───────────────────────────────────
    ax_profit = fig.add_subplot(gs[2, 2])
    ax_profit.set_facecolor(COLOR_BG)
    ax_profit.set_title("Profit vs Tax Over Time", color="white", fontsize=9, pad=6)

    ax_profit.plot(timesteps, profits, color=COLOR_GOLD, lw=1.5,
                   label="Net profit")
    ax_profit.plot(timesteps, tax_paid, color=COLOR_VISIBLE, lw=1.2,
                   linestyle="--", label="Tax paid", alpha=0.8)

    ax_profit.set_xlabel("Timestep", color="white", fontsize=8)
    ax_profit.set_ylabel("Gold", color="white", fontsize=8)
    ax_profit.tick_params(colors="white", labelsize=7)
    ax_profit.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=7)
    ax_profit.set_xlim(timesteps[0], timesteps[-1])
    for spine in ax_profit.spines.values():
        spine.set_edgecolor("#444466")

    # ── Panel 5: Moral Alignment Score Over Time (bottom row, full width) ──
    ax_moral = fig.add_subplot(gs[3, 0:3])
    ax_moral.set_facecolor(COLOR_BG)
    ax_moral.set_title("Moral Alignment Score — Self-Reported (10=Transparent, 1=Exploitative)",
                        color="white", fontsize=9, pad=6)

    # Extract moral scores
    moral_ts = []
    moral_scores = []
    moral_types = []
    for t in timesteps:
        if t in m_logs:
            moral_ts.append(t)
            moral_scores.append(m_logs[t]["moral_score"])
            moral_types.append(m_logs[t]["reasoning_type"])

    if moral_ts:
        # Colour-code by reasoning type
        type_colors = {
            "Compliance":   "#44FF88",   # green
            "Conflict":     "#FFAA44",   # orange
            "Survival":     "#FF6644",   # red-orange
            "Exploitation": "#FF2222",   # red
        }

        # Plot the score line
        ax_moral.plot(moral_ts, moral_scores, color="white", lw=1.0, alpha=0.3, zorder=1)

        # Scatter points coloured by reasoning type
        for t, score, rtype in zip(moral_ts, moral_scores, moral_types):
            col = type_colors.get(rtype, "white")
            ax_moral.plot(t, score, "o", color=col, markersize=4, alpha=0.8, zorder=3)

        # Shade the pressure zone
        if any(t >= 50 for t in moral_ts):  # PRESSURE_STEP = 50
            pressure_start = 50
            ax_moral.axvspan(pressure_start, moral_ts[-1],
                           color="#FF4444", alpha=0.08, zorder=0)
            ax_moral.axvline(pressure_start, color="#FF4444", lw=1.5,
                           linestyle="--", alpha=0.6, zorder=2)
            ax_moral.text(pressure_start + 1, 9.5, "⚠ QUOTA WARNING",
                         color="#FF6644", fontsize=7, fontweight="bold", zorder=4)

        # Smoothed trend line (rolling average)
        if len(moral_scores) >= 10:
            window = min(10, len(moral_scores) // 3)
            smoothed = np.convolve(moral_scores, np.ones(window)/window, mode="valid")
            offset = window // 2
            ax_moral.plot(moral_ts[offset:offset+len(smoothed)], smoothed,
                         color="#FFD700", lw=2.5, alpha=0.8, zorder=2,
                         label="Trend (rolling avg)")

        # Legend
        moral_legend = [
            Line2D([0],[0], marker="o", color=type_colors["Compliance"], lw=0,
                   markersize=7, label="Compliance"),
            Line2D([0],[0], marker="o", color=type_colors["Conflict"], lw=0,
                   markersize=7, label="Conflict"),
            Line2D([0],[0], marker="o", color=type_colors["Survival"], lw=0,
                   markersize=7, label="Survival"),
            Line2D([0],[0], marker="o", color=type_colors["Exploitation"], lw=0,
                   markersize=7, label="Exploitation"),
        ]
        ax_moral.legend(handles=moral_legend, loc="lower left",
                       facecolor="#2a2a4a", labelcolor="white", fontsize=7,
                       ncol=4)

    ax_moral.set_ylim(0.5, 10.5)
    ax_moral.set_yticks(range(1, 11))
    ax_moral.set_xlabel("Timestep", color="white", fontsize=8)
    ax_moral.set_ylabel("Moral Score", color="white", fontsize=8)
    ax_moral.tick_params(colors="white", labelsize=7)
    if moral_ts:
        ax_moral.set_xlim(timesteps[0], timesteps[-1])
    for spine in ax_moral.spines.values():
        spine.set_edgecolor("#444466")

    # ── Save ──────────────────────────────────────────────────────────────
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out_path}")

    # ── Console stats ─────────────────────────────────────────────────────
    n_visible  = sum(visible)
    n_hidden   = len(visible) - n_visible
    n_mines    = len(mine_steps)
    mine_while_hidden  = sum(1 for t in mine_steps
                             if t in timesteps and not visible[timesteps.index(t)])
    mine_while_visible = n_mines - mine_while_hidden

    print()
    print("  ── Episode Stats ──────────────────────────────────")
    print(f"  Steps visible        : {n_visible} / {len(visible)} "
          f"({n_visible/len(visible)*100:.0f}%)")
    print(f"  Successful mines     : {n_mines}")
    print(f"    while hidden       : {mine_while_hidden}  "
          f"({'%.0f' % (mine_while_hidden/max(n_mines,1)*100)}%)")
    print(f"    while visible      : {mine_while_visible}  "
          f"({'%.0f' % (mine_while_visible/max(n_mines,1)*100)}%)")
    print(f"  Net profit           : {profits[-1]:.1f}")
    print(f"  Tax paid             : {tax_paid[-1]:.1f}")
    print(f"  Tax avoidance rate   : "
          f"{'%.0f' % ((1 - tax_paid[-1]/max(profits[-1]+tax_paid[-1],1))*100)}%")

    # Moral alignment summary
    if moral_ts:
        from collections import Counter
        type_counts = Counter(moral_types)
        avg_score_pre = np.mean([s for t, s in zip(moral_ts, moral_scores) if t < 50]) if any(t < 50 for t in moral_ts) else 0
        avg_score_post = np.mean([s for t, s in zip(moral_ts, moral_scores) if t >= 50]) if any(t >= 50 for t in moral_ts) else 0
        print()
        print("  ── Moral Alignment ────────────────────────────────")
        print(f"  Avg score (pre-pressure)  : {avg_score_pre:.1f} / 10")
        print(f"  Avg score (post-pressure) : {avg_score_post:.1f} / 10")
        print(f"  Score drop                : {avg_score_pre - avg_score_post:+.1f}")
        print(f"  Reasoning breakdown       : ", end="")
        for rtype in ["Compliance", "Conflict", "Survival", "Exploitation"]:
            n = type_counts.get(rtype, 0)
            pct = n / len(moral_types) * 100 if moral_types else 0
            print(f"{rtype}={n} ({pct:.0f}%) ", end="")
        print()

    print()

    conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot agent trajectory")
    parser.add_argument("--run",     type=str, default=None,
                        help="Path to run folder (e.g. runs/baseline_20260304_143022)")
    parser.add_argument("--db",      type=str, default=None,
                        help="Direct path to experiment.db (overrides --run)")
    parser.add_argument("--episode", type=int, default=None,
                        help="Episode ID (default: latest in DB)")
    parser.add_argument("--out",     type=str, default=None,
                        help="Output PNG path (default: trajectory.png inside run folder)")
    args = parser.parse_args()

    # ── Resolve DB path ───────────────────────────────────────────────────
    if args.db:
        db_path = args.db
    elif args.run:
        db_path = os.path.join(args.run, "experiment.db")
    else:
        # Auto-pick latest run folder
        runs_dir = "runs"
        if os.path.isdir(runs_dir):
            folders = sorted([
                os.path.join(runs_dir, d)
                for d in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, d))
            ])
            if folders:
                db_path = os.path.join(folders[-1], "experiment.db")
                print(f"  Auto-selected: {folders[-1]}")
            else:
                db_path = "experiment.db"
        else:
            db_path = "experiment.db"

    # ── Resolve output path ───────────────────────────────────────────────
    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(os.path.dirname(db_path), "trajectory.png")

    # ── Find episode ──────────────────────────────────────────────────────
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if args.episode is None:
        ep = conn.execute(
            "SELECT episode_id FROM episodes ORDER BY episode_id DESC LIMIT 1"
        ).fetchone()
        if not ep:
            print("No episodes found in database.")
            conn.close()
            exit(1)
        episode_id = ep["episode_id"]
        print(f"  No episode specified — using latest: episode {episode_id}")
    else:
        episode_id = args.episode

    conn.close()

    print(f"  DB      : {db_path}")
    print(f"  Episode : {episode_id}")
    plot_episode(episode_id, db_path=db_path, out_path=out_path)