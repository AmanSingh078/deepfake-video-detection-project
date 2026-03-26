import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUT = r"d:\deepfake project\deepfake video detection project\images"
if not os.path.exists(OUT): os.makedirs(OUT)

DARK, CARD, TEXT, MUTED = '#0D1117', '#161B22', '#E6EDF3', '#8B949E'
BLUE, GREEN, GOLD, RED = '#2F81F7', '#3FB950', '#D29922', '#F85149'

fig, ax = plt.subplots(figsize=(16, 8))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

# Phases of the Project
phases = [
    {"p": "1. PREPARATION", "steps": "DFDC Dataset\nWeight Setup", "c": BLUE},
    {"p": "2. TRAINING", "steps": "3x Model Ensembles\n40 Epochs Study", "c": GOLD},
    {"p": "3. EVALUATION", "steps": "LogLoss Tests\nMetrics Analysis", "c": GREEN},
    {"p": "4. DEPLOYMENT", "steps": "Flask Backend\nFrontend UI", "c": RED}
]

for i, phase in enumerate(phases):
    # Phase Circle
    circle = patches.Circle((3+i*3, 5), 1.2, linewidth=3, edgecolor=phase['c'], facecolor=CARD, zorder=2)
    ax.add_patch(circle)
    # Title
    ax.text(3+i*3, 5.3, phase['p'], color=phase['c'], fontsize=14, fontweight='bold', ha='center')
    # Steps
    ax.text(3+i*3, 4.7, phase['steps'], color=TEXT, fontsize=11, ha='center', va='center')
    # Connecting Arrows
    if i < len(phases)-1:
        ax.annotate('', xy=(3+(i+1)*3 - 1.2, 5), xytext=(3+i*3 + 1.2, 5),
                    arrowprops=dict(facecolor=TEXT, edgecolor=TEXT, shrink=0.05, width=2, headwidth=10))

ax.text(7.5, 7.5, "THE COMPLETE PROJECT LIFECYCLE", color=TEXT, fontsize=22, fontweight='bold', ha='center')
ax.set_xlim(0, 15)
ax.set_ylim(0, 9)
ax.axis('off')

plt.savefig(os.path.join(OUT, 'project_lifecycle.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
print("✅ Project Lifecycle graph saved")
plt.close()

# --- HOW TO USE FLOW ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

usage = ["1. Open Web Browser", "2. Upload Video", "3. Wait for Analysis", "4. Inspect Score", "5. Case Complete"]
for i, u in enumerate(usage):
    rect = patches.FancyBboxPatch((1, 5-i*1), 8, 0.7, boxstyle="round,pad=0.1", color=CARD, ec=BLUE, lw=2)
    ax.add_patch(rect)
    ax.text(5, 5-i*1+0.35, u, color=TEXT, fontsize=14, fontweight='bold', ha='center')

ax.text(5, 6, "HOW TO USE THE APP (SIMPLE STEPS)", color=TEXT, fontsize=18, fontweight='bold', ha='center')
ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')

plt.savefig(os.path.join(OUT, 'how_to_use_flow.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
print("✅ How-to-use graph saved")
plt.close()
