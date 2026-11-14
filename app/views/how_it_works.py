import os

import streamlit as st

from common import (GALLERY_DIR, HEIGHT, MODEL_COLORS, SPLIT, WIDTH, bank_size, box_style, cut_photo, html_block, image_uri,
                    load_gallery, load_limits, load_patch_grids, load_val_scores, read_bytes, saved_heat_uri, split_counts, svg_uri,
                    yolo_box_html)

GOOD_EXAMPLE, DEFECT_EXAMPLE = "Normal/422", "Anomaly/006"
CAMERA_FILE = os.path.join(GALLERY_DIR, "Anomaly_006_camera.jpg")
K_VALUES = [5, 10, 20, 40]
ORDINAL = {1: "highest", 2: "2nd-highest", 3: "3rd-highest", 4: "4th-highest", 5: "5th-highest"}


def window_style(window, photo_size):
    full_width, full_height = photo_size
    left, top, right, bottom = window
    return (f"left:{100 * left / full_width:.2f}%;top:{100 * top / full_height:.2f}%;"
            f"width:{100 * (right - left) / full_width:.2f}%;height:{100 * (bottom - top) / full_height:.2f}%;")


def ring_style(best_patch, grid_shape):
    rows, cols = grid_shape
    row, col = best_patch
    return f"left:{100 * (col + 0.5) / cols:.2f}%;top:{100 * (row + 0.5) / rows:.2f}%;"


def mini_gauge():
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 30">
<line x1="8" y1="15" x2="212" y2="15" stroke="#ECEBE7" stroke-width="7" stroke-linecap="round"/>
<line x1="112" y1="15" x2="212" y2="15" stroke="#F4D3CF" stroke-width="7" stroke-linecap="round"/>
<line x1="112" y1="4" x2="112" y2="26" stroke="#161616" stroke-width="2.5" stroke-linecap="round"/>
<circle cx="172" cy="15" r="7.5" fill="#C8372D" stroke="#fff" stroke-width="2.5"/></svg>"""
    return f'<img src="{svg_uri(svg)}" alt="score above the limit" style="width:100%;height:auto;background:transparent">'


def beeswarm(xs, radius, max_offset):
    placed = []
    step = 2 * radius + 1
    candidates = [0.0]
    level = 1
    while level * step <= max_offset:
        candidates += [-level * step, level * step]
        level += 1
    for px in xs:
        best = None
        for offset in candidates:
            nearest = min([((px - qx) ** 2 + (offset - qy) ** 2) ** 0.5 for qx, qy in placed] or [99.0])
            if nearest >= step:
                best = offset
                break
            if best is None or nearest > best[1]:
                best = (offset, nearest)
        placed.append((px, best if not isinstance(best, tuple) else best[0]))
    return placed


def limit_chart(good, defect, limit, narrow):
    low, high = 1.8, 3.3
    good_name, defect_name = f"{len(good)} good photos", f"{len(defect)} defect photos"
    if narrow:
        width, height, left, right = 360, 304, 8, 352
        zone_top, zone_bottom, radius = 30, 262, 4
        lanes = [(good_name, good, "#2E7D4F", 112, 62, 52), (defect_name, defect, "#C8372D", 222, 20, 190)]
    else:
        width, height, left, right = 900, 214, 120, 880
        zone_top, zone_bottom, radius = 22, 168, 4.5
        lanes = [(good_name, good, "#2E7D4F", 66, 40, None), (defect_name, defect, "#C8372D", 134, 24, None)]

    def x(value):
        return left + (min(max(value, low), high) - low) / (high - low) * (right - left)

    limit_x = x(limit)
    zone_height = zone_bottom - zone_top
    parts = [f'<svg class="chart {"narrow" if narrow else "wide"}" viewBox="0 0 {width} {height}" role="img" aria-label="Validation scores and the limit">',
             f'<rect x="{limit_x:.1f}" y="{zone_top}" width="{right - limit_x:.1f}" height="{zone_height}" rx="8" fill="#FBEAE8"/>',
             f'<rect x="{left}" y="{zone_top}" width="{limit_x - left:.1f}" height="{zone_height}" rx="8" fill="#F3F8F4"/>']
    zone_label_y = zone_bottom - 10 if narrow else zone_top + 18
    parts.append(f'<text class="lab-muted" x="{left + 8}" y="{zone_label_y}">PASS</text>')
    parts.append(f'<text class="lab-muted" x="{right - 8}" y="{zone_label_y}" text-anchor="end">REJECT</text>')
    for name, scores, color, lane_y, max_offset, label_y in lanes:
        if narrow:
            parts.append(f'<text class="lab" x="{right - 8}" y="{label_y}" text-anchor="end">{name}</text>')
        else:
            parts.append(f'<text class="lab" x="0" y="{lane_y + 4}">{name}</text>')
        for (px, offset), score in zip(beeswarm([x(s) for s in scores], radius, max_offset), scores):
            parts.append(f'<circle cx="{px:.1f}" cy="{lane_y + offset:.1f}" r="{radius}" fill="{color}" stroke="#fff" stroke-width="1.5">'
                         f'<title>score {score:.3f}</title></circle>')
    if narrow:
        parts.append(f'<line x1="{limit_x:.1f}" x2="{limit_x:.1f}" y1="{zone_top - 6}" y2="{zone_bottom + 6}" stroke="#161616" stroke-width="2"/>')
        parts.append(f'<text class="lab" x="{limit_x - 4:.1f}" y="{zone_top - 12}" text-anchor="start">limit {limit:.3f}</text>')
        tick_y = zone_bottom + 22
    else:
        parts.append(f'<line x1="{limit_x:.1f}" x2="{limit_x:.1f}" y1="{zone_top - 8}" y2="{zone_bottom + 8}" stroke="#161616" stroke-width="2"/>')
        parts.append(f'<text class="lab" x="{limit_x:.1f}" y="{zone_bottom + 26}" text-anchor="middle">limit {limit:.3f}</text>')
        tick_y = zone_bottom + 26
    for tick in [2.0, 2.5, 3.0]:
        if narrow or abs(x(tick) - limit_x) > 50:
            parts.append(f'<text class="tick" x="{x(tick):.1f}" y="{tick_y}" text-anchor="middle">{tick:.1f}</text>')
    parts.append(f'<text class="lab-muted" x="{right}" y="{height - 4}" text-anchor="end">PatchCore score, split {SPLIT} validation photos</text>')
    parts.append("</svg>")
    return "".join(parts)


def split_row(label, sub, parts):
    total = sum(count for _, count in parts)
    blocks = "".join(f'<div class="{name}" style="flex:{count} 1 0" title="{name} {count}">{count if count / total > 0.03 else ""}</div>'
                     for name, count in parts)
    return f'<div class="split-row"><div class="label">{label}<span>{sub}</span></div><div class="seg">{blocks}</div></div>'


gallery = load_gallery().set_index("name")
good, defect = gallery.loc[GOOD_EXAMPLE], gallery.loc[DEFECT_EXAMPLE]
limits = load_limits()
grids, _ = load_patch_grids()
grid_rows, grid_cols = grids[DEFECT_EXAMPLE].shape
counts = split_counts()
val_good, val_defect = load_val_scores()
rejected_good = sum(score > limits["PatchCore"] for score in val_good)
camera = cut_photo(read_bytes(CAMERA_FILE))
camera_width, camera_height = camera["photo_size"]
gum_left, gum_top, gum_right, gum_bottom = camera["gum_box"]
gum_share = (gum_right - gum_left) * (gum_bottom - gum_top) / (camera_width * camera_height)
kept = bank_size()
kept_text = f"{kept:,} typical patches" if kept else "1% of the patches"

small_good = image_uri(os.path.join(GALLERY_DIR, good["file"]), (464, 304))
small_defect = image_uri(os.path.join(GALLERY_DIR, defect["file"]), (464, 304))
heat_layer = saved_heat_uri(DEFECT_EXAMPLE, limit=limits["PatchCore"])
ring = ring_style(defect["best_patch"], (grid_rows, grid_cols))
label_boxes = "".join(f'<div class="ybox" style="{box_style(*box[:4])}--c:{MODEL_COLORS["YOLO"]};border-style:dashed"></div>'
                      for box in defect["real_boxes"])
output_boxes = yolo_box_html(defect["yolo_boxes"], limit=limits["YOLO"], chips="top")
key = "".join(f'<span><i style="background:{color}"></i>{name}</span>' for name, color in
              [("train: the model learns from it", "#E4E3DE"), ("validation: sets the limit", "#B5B4AE"),
               ("test: the final score, used once", "#161616"),
               (f"left out: the second photo of {counts['left out']} gums that were photographed twice",
                "repeating-linear-gradient(45deg, #E4E3DE 0 3px, #fff 3px 6px)")])
good_parts = [("train", counts["train good"]), ("val", counts["val good"]), ("test", counts["test good"]), ("out", counts["left out"])]
defect_parts = [("train", counts["train defect"]), ("val", counts["val defect"]), ("test", counts["test defect"])]
k_listed = ", ".join(str(k) for k in K_VALUES[:-1]) + f" or {K_VALUES[-1]}"

html_block(f"""
<div class="page">
<div><h1 class="title">How it works</h1>
<div class="lede">From a camera photo to PASS or REJECT, and how the test was kept fair.</div></div>

<div class="steps">
<div class="step"><span class="num">1</span><div class="name">Take a photo</div>
<div class="what">The camera photo is {camera_width} × {camera_height} pixels. The gum fills only {gum_share:.0%} of it.</div>
<div class="pic photo"><div class="inner"><img src="{image_uri(CAMERA_FILE, None)}" alt="camera photo">
<div class="window" style="{window_style(camera['window'], camera['photo_size'])}"></div></div></div></div>
<div class="step"><span class="num">2</span><div class="name">Cut out the gum</div><div class="what">A {WIDTH} × {HEIGHT} window around the gum. Nothing is stretched.</div>
<div class="pic crop"><img src="{small_defect}" alt="cut out gum"></div></div>
<div class="step"><span class="num">3</span><div class="name">Each model gives a score</div><div class="what">One number per photo. Higher means more likely a defect.</div>
<div class="pic crop"><img src="{small_defect}" alt="photo"><img src="{heat_layer}" alt="heatmap" style="position:absolute;inset:0"><div class="ring" style="{ring}"></div></div></div>
<div class="step"><span class="num">4</span><div class="name">Compare with the limit</div><div class="what">A score above the model's limit means REJECT. Otherwise PASS.</div>
<div class="pic white">{mini_gauge()}<div style="display:flex;justify-content:space-between;font-size:11px;color:#74736F"><span>pass</span><span style="color:#C8372D">reject</span></div>
<div style="text-align:center;margin-top:4px"><span class="pill reject">REJECT</span></div></div></div>
</div>

<div><div class="section-title">Two ways to learn</div>
<div class="section-sub">Both models give one score per photo, but they learn in very different ways.</div>
<div class="grid2">

<div class="card model-card"><div class="head"><i class="dot" style="width:10px;height:10px;background:{MODEL_COLORS['PatchCore']}"></i><span class="name">PatchCore</span><span class="tagline">learns from good photos only</span></div>
<div><div class="pair"><div class="frame"><img src="{small_good}" alt="good gum"><div class="patch-grid"></div></div>
<div class="arrow">→</div>
<div class="frame"><img src="{small_defect}" alt="defect gum"><img src="{heat_layer}" alt="heatmap"><div class="ring" style="{ring}"></div></div></div>
<div class="pair caps" style="margin-top:8px"><div class="cap">Learns: good gum, seen as small patches</div><div></div><div class="cap">Checks: unusual patches turn red</div></div></div>
<ul class="howlist"><li><span class="n">1</span><span>Every photo is cut into {grid_rows * grid_cols:,} small patches ({grid_rows} × {grid_cols}).</span></li>
<li><span class="n">2</span><span>It remembers {kept_text} of the {counts['train good']} good train photos.</span></li>
<li><span class="n">3</span><span>A patch of a new photo that is far from every remembered patch is unusual.</span></li></ul>
<div class="score-line">Score = the most unusual patch <span class="mono" style="margin-left:auto;color:#74736F">{DEFECT_EXAMPLE}: {defect['patchcore_score']:.2f}</span></div></div>

<div class="card model-card"><div class="head"><i class="dot" style="width:10px;height:10px;background:{MODEL_COLORS['YOLO']}"></i><span class="name">YOLO</span><span class="tagline">learns from marked defects</span></div>
<div><div class="pair"><div class="frame"><img src="{small_defect}" alt="marked defects">{label_boxes}</div>
<div class="arrow">→</div>
<div class="frame"><img src="{small_defect}" alt="YOLO boxes">{output_boxes}</div></div>
<div class="pair caps" style="margin-top:8px"><div class="cap">Learns: a box around every defect</div><div></div><div class="cap">Checks: draws boxes with a confidence</div></div></div>
<ul class="howlist"><li><span class="n">1</span><span>Each defect is marked with a box, made from the defect masks of VisA.</span></li>
<li><span class="n">2</span><span>It learns from the {counts['train good']} good photos plus {k_listed} marked defect photos.</span></li>
<li><span class="n">3</span><span>On a new photo it draws boxes, each with a confidence from 0 to 1.</span></li></ul>
<div class="score-line">Score = the most confident box <span class="mono" style="margin-left:auto;color:#74736F">{DEFECT_EXAMPLE}: {defect['yolo_score']:.2f}</span></div></div>
</div></div>

<div class="card"><div class="card-title">Setting the limit</div>
<div class="card-sub">The limit is the {ORDINAL[rejected_good + 1]} score of the {len(val_good)} good validation photos, so at most {rejected_good} of them
({100 * rejected_good / len(val_good):.0f}%) are rejected by mistake. Every model and split gets its own limit this way, before the test photos are used.</div>
<div style="margin-top:14px">{limit_chart(val_good, val_defect, limits["PatchCore"], False)}{limit_chart(val_good, val_defect, limits["PatchCore"], True)}</div></div>

<div class="card"><div class="card-title">A fair test</div>
<div class="card-sub">Every photo is in exactly one part. The test part is used once, for the final score.</div>
{split_row("Good gum", f"{sum(count for _, count in good_parts)} photos", good_parts)}
{split_row("Gum with a defect", f"{sum(count for _, count in defect_parts)} photos", defect_parts)}
<div class="split-key">{key}</div>
<div class="rules">
<div class="rule"><span class="material">repeat</span><b>Done 3 times</b>Three different random splits, so one lucky split cannot decide the result.</div>
<div class="rule"><span class="material">lock</span><b>Test photos stay hidden</b>No setting is chosen with them. They only give the final score.</div>
<div class="rule"><span class="material">filter_none</span><b>No gum twice</b>{counts['left out']} gums were photographed twice. The second photo is left out, so the same gum can never be in train and test at once.</div>
</div></div>
</div>
""")

st.html(f"""<div class="footnote">Pictures and numbers from split {SPLIT}: PatchCore and YOLO with {K_VALUES[-1]} labeled photos.</div>""")
