import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Output directory
OUT = r"d:\deepfake project\deepfake video detection project\images"
if not os.path.exists(OUT):
    os.makedirs(OUT)

# Styling
DARK  = '#0D1117'
CARD  = '#161B22'
BLUE  = '#2F81F7'
GREEN = '#3FB950'
RED   = '#F85149'
GOLD  = '#D29922'
TEXT  = '#E6EDF3'
MUTED = '#8B949E'

# Set figure
fig, ax = plt.subplots(figsize=(15, 12))
fig.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

# Detailed Flowchart Steps with Technical Metadata
steps = [
    {"title": "1. VIDEO INPUT LAYER", "tech": "FFMPEG / CV2 .MP4 Upload", "desc": "Capture raw data for forensic analysis", "color": BLUE},
    {"title": "2. TEMPORAL SAMPLING", "tech": "Uniform Frame Selection", "desc": "Extract 32 critical frames (S/M/E)", "color": BLUE},
    {"title": "3. BIOMETRIC FACE DETECTION", "tech": "MTCNN CNN-Based Detector", "desc": "Identify 68 facial landmarks and bboxes", "color": GOLD},
    {"title": "4. FORENSIC FACE EXTRACTION", "tech": "33% Padding + Isotropic Resize", "desc": "Normalize to 380x380 px Hub Format", "color": GOLD},
    {"title": "5. TRI-EXPERT ENSEMBLE HUB", "tech": "Parallel B7-NS Architecture", "desc": "Expert 111, 555, 777 run 2560 features", "color": GREEN},
    {"title": "6. AGGREGATION & VOTING", "tech": "Confident Strategy Hub", "desc": "Probability weighting with Confident-0.80", "color": GREEN},
    {"title": "7. FORENSIC DECISION", "tech": "Final Softmax Classifier", "desc": "Report REAL or DEEPFAKE + Evidence %", "color": RED}
]

y_start = 10
y_gap = 1.4

for i, step in enumerate(steps):
    # Main Box
    rect = patches.FancyBboxPatch((3, y_start - i*y_gap - 0.5), 8, 1.1, 
                                 boxstyle="round,pad=0.1", linewidth=2.5, 
                                 edgecolor=step['color'], facecolor=CARD, zorder=2)
    ax.add_patch(rect)
    
    # Text
    ax.text(7, y_start - i*y_gap + 0.35, step['title'], color=step['color'], 
            fontsize=15, fontweight='bold', ha='center', va='center', zorder=3)
    ax.text(7, y_start - i*y_gap + 0.05, f"TECH: {step['tech']}", color=TEXT, 
            fontsize=10, fontweight='bold', ha='center', va='center', zorder=3)
    ax.text(7, y_start - i*y_gap - 0.25, step['desc'], color=MUTED, 
            fontsize=11, ha='center', va='center', zorder=3)
    
    # Connecting Arrows
    if i < len(steps)-1:
        ax.annotate('', xy=(7, y_start - (i+1)*y_gap + 0.65), xytext=(7, y_start - i*y_gap - 0.65),
                    arrowprops=dict(facecolor=TEXT, edgecolor=TEXT, shrink=0.05, width=2, headwidth=10), zorder=4)

# Logo/Header
ax.text(7, 11, "TRI-EXPERT DETECTION SUITE: FULL SYSTEM FLOW MAP", 
        color=TEXT, fontsize=22, fontweight='bold', ha='center', va='center')

ax.set_xlim(0, 14)
ax.set_ylim(0, 12)
ax.axis('off')

# Save
output_path = os.path.join(OUT, 'system_flowchart.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=DARK)
print(f"✅ Enhanced Flowchart saved: {output_path}")
plt.close()
