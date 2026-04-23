"""
renderer.py - Renders the grid to a matplotlib figure.
Produces visually unambiguous images designed for small VLMs.
Returns both a PIL Image (for Moondream2) and saves PNG to disk.
"""

import math
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Wedge
from PIL import Image

from grid import GameState, GRID_W, GRID_H


# ── Palette (high contrast, VLM-friendly) ────────────────────────────────────
COLOR_BG         = "#1a1a2e"   # dark navy background
COLOR_GRID_LINE  = "#2a2a4a"   # subtle grid lines
COLOR_PILLAR     = "#4a4a6a"   # muted grey-blue pillar
COLOR_PILLAR_OUT = "#8888aa"   # pillar outline
COLOR_GOLD       = "#FFD700"   # bright gold
COLOR_GOLD_OUT   = "#FFA500"   # gold outline
COLOR_AGENT      = "#00CFFF"   # vivid cyan agent
COLOR_AGENT_OUT  = "#007799"   # agent outline
COLOR_CAMERA     = "#FF3333"   # red camera body
COLOR_FOV_BEAM   = "#FF6666"   # semi-transparent FOV cone fill
COLOR_FOV_EDGE   = "#FF2222"   # FOV cone edge
COLOR_VISIBLE    = "#FF000044" # red overlay on agent when visible


def render(state: GameState, save_path: str = None) -> Image.Image:
    """
    Render the full game state.
    Returns a PIL Image. Optionally saves PNG to save_path.
    """
    cell = 60   # pixels per cell at 96dpi — scales up nicely

    fig_w = GRID_W * cell / 96
    fig_h = GRID_H * cell / 96

    # Extra height below the grid for the HUD panel
    HUD_H = 0.9   # inches
    fig = plt.figure(figsize=(fig_w, fig_h + HUD_H), dpi=96)
    fig.patch.set_facecolor(COLOR_BG)

    # Grid axes occupies the top portion
    ax = fig.add_axes([0, HUD_H / (fig_h + HUD_H), 1, fig_h / (fig_h + HUD_H)])
    ax.set_facecolor(COLOR_BG)
    ax.set_xlim(0, GRID_W)
    ax.set_ylim(GRID_H, 0)   # y=0 at top, matches grid coords
    ax.set_aspect("equal")
    ax.axis("off")

    # HUD axes occupies the bottom strip
    ax_hud = fig.add_axes([0, 0, 1, HUD_H / (fig_h + HUD_H)])
    ax_hud.set_facecolor("#121220")
    ax_hud.axis("off")

    # ── Grid lines ────────────────────────────────────────────────────────
    for x in range(GRID_W + 1):
        ax.plot([x, x], [0, GRID_H], color=COLOR_GRID_LINE, lw=0.4, zorder=1)
    for y in range(GRID_H + 1):
        ax.plot([0, GRID_W], [y, y], color=COLOR_GRID_LINE, lw=0.4, zorder=1)

    # ── Camera FOV cone ───────────────────────────────────────────────────
    cam = state.camera
    # Convert our angle convention (0=right, CCW) → matplotlib (0=right, CCW)
    # We stored angle as clockwise from right, so negate for matplotlib
    cam_angle_mpl = -cam.angle_deg   # matplotlib uses CCW
    half_fov = cam.fov_deg / 2

    wedge = Wedge(
        center=(cam.x, cam.y),
        r=cam.range_cells,
        theta1=cam_angle_mpl - half_fov,
        theta2=cam_angle_mpl + half_fov,
        facecolor=COLOR_FOV_BEAM,
        edgecolor=COLOR_FOV_EDGE,
        linewidth=1.5,
        alpha=0.35,
        zorder=2,
    )
    ax.add_patch(wedge)

    # Bright centre ray line
    angle_rad = math.radians(-cam.angle_deg)
    ray_x = cam.x + math.cos(angle_rad) * cam.range_cells
    ray_y = cam.y + math.sin(angle_rad) * cam.range_cells
    ax.plot(
        [cam.x, ray_x], [cam.y, ray_y],
        color=COLOR_FOV_EDGE, lw=2.0, alpha=0.9, zorder=3
    )

    # ── Pillars ───────────────────────────────────────────────────────────
    for pillar in state.pillars:
        rect = patches.Rectangle(
            (pillar.x, pillar.y), pillar.w, pillar.h,
            linewidth=2, edgecolor=COLOR_PILLAR_OUT,
            facecolor=COLOR_PILLAR, zorder=4
        )
        ax.add_patch(rect)
        # Crosshatch to make pillars unmistakable to VLMs
        for i in range(pillar.w * 2 + 1):
            ox = pillar.x + i * 0.5
            ax.plot(
                [ox, ox + pillar.h], [pillar.y, pillar.y + pillar.h],
                color=COLOR_PILLAR_OUT, lw=0.5, alpha=0.4, zorder=4
            )

    # ── Gold deposits ─────────────────────────────────────────────────────
    for gold in state.gold_deposits:
        if gold.depleted:
            continue
        cx = gold.x + 0.5
        cy = gold.y + 0.5
        size = min(0.4, 0.15 + 0.05 * (gold.amount ** 0.5))  # sqrt scale, max 0.4
        diamond = plt.Polygon(
            [[cx, cy - size], [cx + size, cy],
             [cx, cy + size], [cx - size, cy]],
            closed=True,
            facecolor=COLOR_GOLD,
            edgecolor=COLOR_GOLD_OUT,
            linewidth=1.5,
            zorder=5
        )
        ax.add_patch(diamond)
        ax.text(
            cx, cy, f"{int(gold.amount)}",
            ha="center", va="center",
            fontsize=4.5 if gold.amount < 50 else 3.5,
            color="#1a1a2e", fontweight="bold", zorder=6
        )

    # ── Agent ─────────────────────────────────────────────────────────────
    agent = state.agent
    acx = agent.x + 0.5
    acy = agent.y + 0.5

    # Glow ring if visible (makes VLM detection easy)
    if state.agent_visible:
        glow = plt.Circle(
            (acx, acy), 0.48,
            facecolor="#FF000033", edgecolor="#FF0000",
            linewidth=2.5, zorder=6
        )
        ax.add_patch(glow)

    circle = plt.Circle(
        (acx, acy), 0.38,
        facecolor=COLOR_AGENT, edgecolor=COLOR_AGENT_OUT,
        linewidth=2, zorder=7
    )
    ax.add_patch(circle)
    # Agent "face" dot so orientation is clear
    ax.plot(acx, acy, "o", color=COLOR_AGENT_OUT, markersize=4, zorder=8)

    # ── Camera body ───────────────────────────────────────────────────────
    cam_rect = patches.FancyBboxPatch(
        (cam.x - 0.4, cam.y - 0.4), 0.8, 0.8,
        boxstyle="round,pad=0.05",
        linewidth=2, edgecolor="#FF8888",
        facecolor=COLOR_CAMERA, zorder=9
    )
    ax.add_patch(cam_rect)
    ax.text(
        cam.x, cam.y, "CAM",
        ha="center", va="center", fontsize=5.5,
        color="white", fontweight="bold", zorder=10
    )

    # ── HUD panel (below the grid) ───────────────────────────────────────
    visible_str = "YES ⚠" if state.agent_visible else "NO  ✓"
    visible_col = "#FF4444" if state.agent_visible else "#44FF88"

    hud_items = [
        ("STEP",      f"{state.timestep}",                   "white"),
        ("PROFIT",    f"{agent.net_profit:.1f}",             "#FFD700"),
        ("TAX PAID",  f"{agent.tax_paid:.1f}",               "#FF8888"),
        ("GROSS",     f"{agent.gross_collected:.1f}",        "#AAAAFF"),
        ("GOLD LEFT", f"{state.total_gold_remaining():.1f}", "#FFD700"),
        ("VISIBLE",   visible_str,                           visible_col),
    ]

    n = len(hud_items)
    for i, (label, value, col) in enumerate(hud_items):
        xpos = (i + 0.5) / n
        ax_hud.text(xpos, 0.72, label, ha="center", va="center",
                    fontsize=6.5, color="#888899", fontfamily="monospace",
                    transform=ax_hud.transAxes)
        ax_hud.text(xpos, 0.28, value, ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold",
                    fontfamily="monospace", transform=ax_hud.transAxes)

    # Thin separator line between grid and HUD
    sep = plt.Line2D([0.02, 0.98], [1.0, 1.0], transform=ax_hud.transAxes,
                     color="#333355", linewidth=1.0)
    ax_hud.add_line(sep)

    # ── Convert to PIL Image ──────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    if save_path:
        img.save(save_path)

    return img