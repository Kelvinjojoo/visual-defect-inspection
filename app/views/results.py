import os

import pandas as pd
import streamlit as st

from common import (GALLERY_DIR, MODEL_COLORS, POINTING, RESULTS_DIR, TIME_MS, html_block, image_uri, load_gallery,
                    load_test_results, pct, real_box_html)

CAUGHT_LABELS = [("YOLO", 5, "below"), ("YOLO", 40, "above"), ("PatchCore", None, "above")]
REJECTED_LABELS = [("YOLO", 10, "above"), ("YOLO", 40, "below"), ("PatchCore", None, "above")]
MISSED = {"PatchCore": "Anomaly/032", "YOLO": "Anomaly/026"}
MISSED_WHY = {"PatchCore": "tiny pieces broken off the corners. The photo shows a bit more mat, and mat looks normal.",
              "YOLO": "a bigger missing corner, with 40 labeled photos."}


@st.cache_data
def load_decisions():
    return pd.read_csv(os.path.join(RESULTS_DIR, "evaluation", "test_decisions.csv")).fillna({"defect_types": ""})


results = load_test_results()
SPLITS = sorted(results["split"].unique())
K_VALUES = sorted(results.loc[results["k"] > 0, "k"].unique())
BEST_K = max(K_VALUES)
ROWS = []
for k, rows in results.groupby("k"):
    ROWS.append({"model": "PatchCore" if k == 0 else "YOLO", "k": int(k), "caught": int(rows["caught"].sum()),
                 "rejected": int(rows["false rejects"].sum()), "auroc": rows["AUROC"].mean(),
                 "caught per split": rows.sort_values("split")["caught"].tolist()})
DEFECT_PHOTOS = int(results.loc[results["k"] == 0, "defect photos"].sum())
GOOD_PHOTOS = int(results.loc[results["k"] == 0, "normal photos"].sum())
patchcore = next(row for row in ROWS if row["k"] == 0)
yolo = {row["k"]: row for row in ROWS if row["k"] > 0}
best = yolo[BEST_K]

decisions = load_decisions()
pair = decisions[decisions["model"].isin(["PatchCore", f"YOLO k = {BEST_K}"])]
defect_votes = pair[pair["is_defect"]].pivot_table(index=["split", "image"], columns="model", values="rejected", aggfunc="first")
missed_by_both = int((~defect_votes.any(axis=1)).sum())
missed = pair[pair["is_defect"] & ~pair["rejected"]]
all_corners = bool(missed["defect_types"].str.contains("corner missing").all())


def line_chart(title, values_of, y_min, y_max, ticks, label_rules, width, size_class):
    height = 262
    left, right, top, bottom = 44, width - 34, 16, 206
    xs = [left + i * (right - left) / len(K_VALUES) for i in range(len(K_VALUES) + 1)]

    def y(value):
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [f'<svg class="chart {size_class}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">']
    for tick in ticks:
        parts.append(f'<line x1="{left}" x2="{right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#ECEBE7" stroke-width="1"/>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y(tick) + 4:.1f}" text-anchor="end">{tick:g}%</text>')
    for x, k in zip(xs, [0] + K_VALUES):
        parts.append(f'<text class="tick" x="{x:.1f}" y="{bottom + 22}" text-anchor="middle">{k}</text>')
    parts.append(f'<text class="lab-muted" x="{(left + right) / 2:.1f}" y="{bottom + 48}" text-anchor="middle">labeled defect photos</text>')
    patchcore_value = values_of(patchcore)
    patchcore_y = y(patchcore_value)
    parts.append(f'<line x1="{xs[0]:.1f}" x2="{xs[-1]:.1f}" y1="{patchcore_y:.1f}" y2="{patchcore_y:.1f}" '
                 f'stroke="{MODEL_COLORS["PatchCore"]}" stroke-width="2" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{xs[0]:.1f}" cy="{patchcore_y:.1f}" r="5" fill="{MODEL_COLORS["PatchCore"]}" stroke="#fff" stroke-width="2">'
                 f'<title>PatchCore, no labeled photos: {patchcore_value:.1f}%</title></circle>')
    points = [(xs[i + 1], y(values_of(yolo[k])), k, values_of(yolo[k])) for i, k in enumerate(K_VALUES)]
    path = " ".join(f"{'M' if i == 0 else 'L'}{px:.1f},{py:.1f}" for i, (px, py, _, _) in enumerate(points))
    parts.append(f'<path d="{path}" fill="none" stroke="{MODEL_COLORS["YOLO"]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for px, py, k, value in points:
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{MODEL_COLORS["YOLO"]}" stroke="#fff" stroke-width="2">'
                     f'<title>YOLO, {k} labeled photos: {value:.1f}%</title></circle>')
    for model, k, place in label_rules:
        if model == "PatchCore":
            px, py, value = xs[0], patchcore_y, patchcore_value
        else:
            px, py, _, value = next(point for point in points if point[2] == k)
        dy = -12 if place == "above" else 22
        anchor = "end" if px > right - 20 else ("start" if px < left + 20 else "middle")
        shift = {"end": 6, "start": 8, "middle": 0}[anchor]
        parts.append(f'<text class="lab" x="{px + shift:.1f}" y="{py + dy:.1f}" text-anchor="{anchor}">{value:.1f}%</text>')
    parts.append("</svg>")
    return "".join(parts)


def split_line(split, difference):
    if difference == 0:
        return f'<div><span class="s">Split {split}</span><i class="dot" style="background:#C9C8C3"></i><span class="tie">same number</span></div>'
    winner = "PatchCore" if difference > 0 else "YOLO"
    return f'<div><span class="s">Split {split}</span><i class="dot" style="background:{MODEL_COLORS[winner]}"></i>{winner}, {abs(difference)} more</div>'


def k_card(k):
    differences = [p - y for p, y in zip(patchcore["caught per split"], yolo[k]["caught per split"])]
    if all(d > 0 for d in differences):
        verdict, style = "PatchCore clearly catches more", ""
    elif all(d < 0 for d in differences):
        verdict, style = "YOLO clearly catches more", ""
    else:
        verdict, style = "No clear winner", " none"
    lines = "".join(split_line(split, d) for split, d in zip(SPLITS, differences))
    return f"""<div class="k-card"><div class="k">{k} labeled defect photos</div>
<div class="verdict-text{style}">{verdict}</div><div class="splits">{lines}</div></div>"""


def miss_card(model, gallery):
    row = gallery.set_index("name").loc[MISSED[model]]
    return f"""<div class="miss"><div class="frame"><img src="{image_uri(os.path.join(GALLERY_DIR, row['file']), (464, 304))}" alt="{MISSED[model]}">
{real_box_html(row['real_boxes'])}</div>
<div class="who"><i class="dot" style="background:{MODEL_COLORS[model]}"></i>Missed by {model}</div>
<div class="why"><span class="mono">{MISSED[model]}</span>: {MISSED_WHY[model]}</div></div>"""


caught_chart = "".join(line_chart("Defect photos caught", lambda row: pct(row["caught"], DEFECT_PHOTOS), 80, 100, [80, 85, 90, 95, 100],
                                  CAUGHT_LABELS, width, size_class) for width, size_class in [(520, "wide"), (380, "narrow")])
rejected_chart = "".join(line_chart("Good photos rejected", lambda row: pct(row["rejected"], GOOD_PHOTOS), 0, 12, [0, 4, 8, 12],
                                    REJECTED_LABELS, width, size_class) for width, size_class in [(520, "wide"), (380, "narrow")])
key = (f'<div class="key"><span><i class="line" style="background:{MODEL_COLORS["PatchCore"]}"></i>PatchCore, needs no labeled photos</span>'
       f'<span><i class="line" style="background:{MODEL_COLORS["YOLO"]}"></i>YOLO</span></div>')
k_cards = "".join(k_card(k) for k in K_VALUES)

compare_rows = [
    ("Labeled defect photos needed", "0", f"{BEST_K}", "PatchCore"),
    ("Defect photos caught", f"{patchcore['caught']} of {DEFECT_PHOTOS}", f"{best['caught']} of {DEFECT_PHOTOS}", None),
    ("Good photos rejected", f"{patchcore['rejected']} of {GOOD_PHOTOS}", f"{best['rejected']} of {GOOD_PHOTOS}", "YOLO"),
    ("AUROC (1.0 is perfect)", f"{patchcore['auroc']:.3f}", f"{best['auroc']:.3f}", "YOLO"),
    ("Points at the real defect", f"{POINTING['PatchCore']}%", f"{POINTING['YOLO']}%", None),
    ("Time per photo, GPU", f"{TIME_MS['PatchCore']['GPU']} ms", f"{TIME_MS['YOLO']['GPU']} ms", "YOLO"),
    ("Time per photo, CPU", f"{TIME_MS['PatchCore']['CPU']} ms", f"{TIME_MS['YOLO']['CPU']} ms", "YOLO"),
]
table_rows = "".join(f"""<tr><td>{name}</td><td class="{'better' if better == 'PatchCore' else ''}">{a}</td>
<td class="{'better' if better == 'YOLO' else ''}">{b}</td></tr>""" for name, a, b, better in compare_rows)
all_rows = "".join(f"""<tr><td>{row['model']}</td><td>{row['k']}</td>
<td>{row['caught']} of {DEFECT_PHOTOS} ({pct(row['caught'], DEFECT_PHOTOS):.1f}%)</td>
<td>{row['rejected']} of {GOOD_PHOTOS} ({pct(row['rejected'], GOOD_PHOTOS):.1f}%)</td><td>{row['auroc']:.3f}</td></tr>""" for row in ROWS)

gallery = load_gallery()
miss_sub = "Every missed photo has a missing corner, the hardest defect type." if all_corners else "The photos each model missed."
if missed_by_both == 0:
    miss_note = (f"Each model catches the photos the other one misses. With {BEST_K} labeled photos, "
                 "every defect test photo is caught by at least one of the two.")
else:
    miss_note = f"{missed_by_both} defect test photos are missed by both models."

html_block(f"""
<div class="page">
<div><h1 class="title">Results</h1>
<div class="lede">How both models did on all test photos: {DEFECT_PHOTOS} defect photos and {GOOD_PHOTOS} good photos from {len(SPLITS)} splits.
Each model saw them once.</div></div>

<div class="answer-card"><div class="q">Does a factory need labeled defects?</div>
<div class="a">Not to catch most defects. PatchCore learned from good photos only and still caught {pct(patchcore['caught'], DEFECT_PHOTOS):.1f}% of them.
YOLO needed {BEST_K} labeled defect photos to catch as many, and then it threw away less good gum.</div>
<div class="facts">
<div class="fact"><span class="who"><i class="dot" style="background:{MODEL_COLORS['PatchCore']}"></i>PatchCore, 0 labeled photos</span>
<span class="big">{pct(patchcore['caught'], DEFECT_PHOTOS):.1f}% caught</span>{pct(patchcore['rejected'], GOOD_PHOTOS):.1f}% of good gum rejected</div>
<div class="fact"><span class="who"><i class="dot" style="background:{MODEL_COLORS['YOLO']}"></i>YOLO, {BEST_K} labeled photos</span>
<span class="big">{pct(best['caught'], DEFECT_PHOTOS):.1f}% caught</span>{pct(best['rejected'], GOOD_PHOTOS):.1f}% of good gum rejected</div>
</div></div>

<div class="grid2">
<div class="card chart-card"><div class="card-title">Defect photos caught</div><div class="card-sub">All {len(SPLITS)} splits together. Higher is better.</div>{key}{caught_chart}</div>
<div class="card chart-card"><div class="card-title">Good photos rejected by mistake</div><div class="card-sub">All {len(SPLITS)} splits together. Lower is better.</div>{key}{rejected_chart}</div>
</div>

<div><div class="section-title">At each amount of labels</div>
<div class="section-sub">Who caught more defect photos in each split ({DEFECT_PHOTOS // len(SPLITS)} defect photos per split).
A model clearly catches more only if it wins all {len(SPLITS)} splits.</div>
<div class="grid4">{k_cards}</div></div>

<div class="grid2">
<div class="card"><div class="card-title">PatchCore vs YOLO with {BEST_K} labeled photos</div><div class="card-sub">The closest match. Bold is clearly better.</div>
<table class="compare" style="margin-top:16px"><tr><th>Measure</th><th><span><i class="dot" style="background:{MODEL_COLORS['PatchCore']}"></i>PatchCore</span></th>
<th><span><i class="dot" style="background:{MODEL_COLORS['YOLO']}"></i>YOLO</span></th></tr>{table_rows}</table></div>

<div class="card"><div class="card-title">They miss different photos</div><div class="card-sub">{miss_sub}</div>
<div class="miss-grid">{miss_card("PatchCore", gallery)}{miss_card("YOLO", gallery)}</div>
<div class="note">{miss_note}</div></div>
</div>

<details class="table-view"><summary>Show all numbers as a table</summary>
<table class="all"><tr><th>Model</th><th>Labeled defect photos</th><th>Defect photos caught</th><th>Good photos rejected</th><th>AUROC</th></tr>{all_rows}</table></details>
</div>
""")

st.html(f"""<div class="footnote">Numbers from notebooks 5.1 to 5.3. The test photos of the {len(SPLITS)} splits together;
percentages and AUROC are the means of the {len(SPLITS)} splits. Dashed boxes are the real defects.</div>""")
