import math
import os

import streamlit as st
from PIL import Image

import inference
from common import (CHECK, CROSS, GALLERY_DIR, HEAT_STRENGTH, HEIGHT, K, SPLIT, TEST_PHOTO_DIR, TIME_MS, WIDTH, cut_photo, encode,
                    heat_uri, image_uri, load_gallery, load_limits, load_new_photos, load_patch_grids, load_split, read_bytes,
                    real_box_html, saved_heat_uri, split_counts, svg_uri, yolo_box_html)

COLLECTIONS = ["Test photos", "Hard cases", "New photos"]
CARD_TEXT = {"Test photos": ("Test photos of split 1", "Neither model saw these photos while learning."),
             "Hard cases": ("Hard cases of split 1", "Test photos that one of the two models got wrong."),
             "New photos": ("New photos", "Full camera photos made for this demo. Both models check them live.")}
DEFAULT_PHOTO = "Anomaly/006"
THUMB_SIZE = (232, 152)
PART_TEXT = {"train": "a train photo of split 1", "val": "a validation photo of split 1", "test": "a test photo of split 1",
             "excluded": "a photo left out in notebook 1.2", "new": "one of the new photos"}


@st.cache_data(show_spinner=False)
def new_thumb(file):
    crop = cut_photo(read_bytes(os.path.join(TEST_PHOTO_DIR, file)))["crop"]
    return encode(crop.resize(THUMB_SIZE, Image.LANCZOS), "JPEG")


@st.cache_data(show_spinner=False, max_entries=32)
def live_check(photo_bytes, heat_low, limit):
    checked = inference.check_photo(photo_bytes)
    cut = cut_photo(photo_bytes)
    return {**checked, "photo_uri": encode(cut["crop"], "JPEG"), "window": cut["window"], "photo_size": cut["photo_size"],
            "heat_uri": heat_uri(checked["grid"], heat_low=heat_low, limit=limit, strength=HEAT_STRENGTH)}


def key_of(photo_id):
    return photo_id.replace("/", "_")


def listed(defect_types):
    types = [name for name in defect_types.split(",") if name]
    return types[0] if len(types) == 1 else ", ".join(types[:-1]) + " and " + types[-1]


def label_text(is_defect, defect_types):
    if is_defect:
        return f'has a defect ({listed(defect_types)}), so the right answer is <b class="reject">REJECT</b>.'
    return 'is good gum, so the right answer is <b class="pass">PASS</b>.'


def notice(text, kind):
    symbol = {"warning": "warning", "info": "info"}[kind]
    st.html(f'<div class="notice {kind}"><span class="material">{symbol}</span><span>{text}</span></div>')


def gauge(score, limit, low, high, log_scale, decision_color):
    def x(value):
        value = min(max(value, low), high)
        if log_scale:
            share = (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
        else:
            share = (value - low) / (high - low)
        return 10 + share * 580
    limit_x, score_x = x(limit), x(score)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 34">
<line x1="10" y1="17" x2="590" y2="17" stroke="#ECEBE7" stroke-width="8" stroke-linecap="round"/>
<line x1="{limit_x:.1f}" y1="17" x2="590" y2="17" stroke="#F4D3CF" stroke-width="8" stroke-linecap="round"/>
<line x1="{limit_x:.1f}" y1="4" x2="{limit_x:.1f}" y2="30" stroke="#161616" stroke-width="2.5" stroke-linecap="round"/>
<circle cx="{score_x:.1f}" cy="17" r="8.5" fill="{decision_color}" stroke="#fff" stroke-width="3"/>
</svg>"""
    return f'<img class="gauge" src="{svg_uri(svg)}" alt="score {score:.3g}, limit {limit:.3g}">'


def panel(name, sub, score, limit, low, high, log_scale, figure, foot, is_defect):
    rejected = score > limit
    decision, color = ("REJECT", "#C8372D") if rejected else ("PASS", "#2E7D4F")
    if is_defect is None:
        mark = ""
    elif rejected == is_defect:
        mark = f'<span class="mark right"><img src="{CHECK}" alt="">right</span>'
    elif is_defect:
        mark = f'<span class="mark wrong"><img src="{CROSS}" alt="">missed the defect</span>'
    else:
        mark = f'<span class="mark wrong"><img src="{CROSS}" alt="">false alarm</span>'
    return f"""<div class="panel">
<div class="panel-head"><div><div class="panel-name">{name}</div><div class="panel-sub">{sub}</div></div>
<div class="decision"><span class="pill {decision.lower()}">{decision}</span>{mark}</div></div>
<div class="score-row"><span class="score">{score:.3g}</span><span class="score-label">score</span>
<span class="score-label" style="margin-left:auto">limit <span class="time">{limit:.3g}</span></span></div>
{gauge(score, limit, low, high, log_scale, color)}
<div class="zones"><span>pass</span><span class="reject">reject</span></div>
{figure}
<div class="panel-foot">{foot}</div>
</div>"""


gallery = load_gallery()
new_photos = load_new_photos()
limits = load_limits()
_, heat_low = load_patch_grids()
counts = split_counts()
ITEMS = {"Test photos": [{"id": row["name"], "is_defect": bool(row["is_defect"]), "defect_types": row["defect_types"],
                          "thumb": image_uri(os.path.join(GALLERY_DIR, row["file"]), THUMB_SIZE)}
                         for _, row in gallery[~gallery["hard_case"]].iterrows()],
         "Hard cases": [{"id": row["name"], "is_defect": bool(row["is_defect"]), "defect_types": row["defect_types"],
                         "thumb": image_uri(os.path.join(GALLERY_DIR, row["file"]), THUMB_SIZE)}
                        for _, row in gallery[gallery["hard_case"]].iterrows()],
         "New photos": [{"id": row["name"], "is_defect": row["right_answer"] == "REJECT", "defect_types": row["defect_types"],
                         "thumb": new_thumb(row["file"])} for _, row in new_photos.iterrows()]}


def pick_photo(photo_id):
    st.session_state.photo = photo_id


def pick_collection():
    st.session_state.photo = ITEMS[st.session_state.collection][0]["id"]


def use_upload():
    if st.session_state.upload is not None:
        st.session_state.photo = "upload"
    elif st.session_state.photo == "upload":
        st.session_state.photo = ITEMS[st.session_state.collection][0]["id"]


if "photo" not in st.session_state:
    st.session_state.photo = DEFAULT_PHOTO
if "collection" not in st.session_state:
    st.session_state.collection = COLLECTIONS[0]

with st.sidebar:
    st.html(f"""<div class="side-card"><h4>Models of split {SPLIT}</h4>
<div class="row"><b>PatchCore</b>Learned from {counts['train good']} good photos. No labeled defects.</div>
<div class="row"><b>YOLO</b>Learned from {counts['train good']} good photos and {counts['labeled']} photos with marked defects.</div>
<div class="row"><b>Limits</b>Set on {counts['val good']} good validation photos (notebook 5.1).</div>
</div>""")

head_text, head_button = st.columns([5, 1.25], vertical_alignment="bottom")
with head_text:
    st.html("""<h1 class="title">Inspect a gum photo</h1>
<p class="lede">Both models judge the same photo. A score above the model's limit means REJECT.</p>""")
with head_button:
    with st.popover("Upload a photo", width="stretch"):
        st.file_uploader("Photo of one piece of gum", type=["jpg", "jpeg", "png"], key="upload", on_change=use_upload)
        st.caption(f"A full camera photo, like the new photos, or a cut-out gum of {WIDTH} × {HEIGHT} pixels. "
                   "The gum is cut out first, as in notebook 3.1.")
        if st.session_state.get("upload") is not None and st.session_state.photo != "upload":
            st.button("Show my photo", on_click=pick_photo, args=("upload",), width="stretch")

st.space("small")
with st.container(key="gallery"):
    title, choose = st.columns([3, 2.2], vertical_alignment="center")
    card_title, card_sub = CARD_TEXT[st.session_state.collection]
    with title:
        st.html(f'<div class="card-title">{card_title}</div><div class="card-sub">{card_sub}</div>')
    with choose:
        st.segmented_control("Show", COLLECTIONS, key="collection", required=True, on_change=pick_collection,
                             label_visibility="collapsed")
    with st.container(horizontal=True, wrap=False, gap="small", key="strip"):
        for item in ITEMS[st.session_state.collection]:
            selected = item["id"] == st.session_state.photo
            tag = item["defect_types"].split(",")[0] if item["is_defect"] else "good gum"
            dot = "#C8372D" if item["is_defect"] else "#2E7D4F"
            with st.container(key=f"thumb_{key_of(item['id'])}_{'on' if selected else 'off'}", width=96):
                st.html(f"""<div class="thumb"><img src="{item['thumb']}" alt="{item['id']}">
<div class="name">{item['id']}</div><div class="tag"><span class="dot" style="background:{dot}"></span><span class="text">{tag}</span></div></div>""")
                st.button(item["id"], key=f"pick_{key_of(item['id'])}", on_click=pick_photo, args=(item["id"],))

photo = st.session_state.photo
if photo == "upload" and st.session_state.get("upload") is None:
    photo = st.session_state.photo = DEFAULT_PHOTO
gallery_row = gallery.set_index("name").loc[photo] if photo in set(gallery["name"]) else None
new_row = new_photos.set_index("name").loc[photo] if photo in set(new_photos["name"]) else None
st.space("small")

if gallery_row is not None:
    view = {"photo_uri": image_uri(os.path.join(GALLERY_DIR, gallery_row["file"]), None),
            "heat_uri": saved_heat_uri(photo, limit=limits["PatchCore"]),
            "patchcore_score": float(gallery_row["patchcore_score"]), "yolo_score": float(gallery_row["yolo_score"]),
            "yolo_boxes": gallery_row["yolo_boxes"], "live": False}
    is_defect, real_boxes = bool(gallery_row["is_defect"]), gallery_row["real_boxes"]
    answer, warning = f'<span class="mono">{photo}</span> {label_text(is_defect, gallery_row["defect_types"])}', None
else:
    if new_row is not None:
        photo_bytes = read_bytes(os.path.join(TEST_PHOTO_DIR, new_row["file"]))
    else:
        photo_bytes = st.session_state.upload.getvalue()
    try:
        cut = cut_photo(photo_bytes)
    except Exception:
        notice("This file could not be opened as a photo. Try a JPG or PNG photo.", kind="warning")
        st.stop()
    photo_width, photo_height = cut["photo_size"]
    if photo_width < WIDTH or photo_height < HEIGHT:
        notice(f"This photo is {photo_width} × {photo_height} pixels. The models need at least {WIDTH} × {HEIGHT} pixels: "
               f"a full camera photo or a cut-out gum of {WIDTH} × {HEIGHT}.", kind="warning")
        st.stop()
    missing = inference.models_missing()
    if missing:
        notice("Live checks need the model files " + " and ".join(f'<span class="mono">{path}</span>' for path in missing)
               + ". Notebooks 4.1 and 4.2 make them.", kind="warning")
        st.stop()
    try:
        with st.spinner("Loading the models. The first time can take up to a minute."):
            inference.load_models()
        with st.spinner("Checking the photo with both models."):
            view = {**live_check(photo_bytes, heat_low=heat_low, limit=limits["PatchCore"]), "live": True}
    except Exception as error:
        notice(f"The models could not check this photo: {error}", kind="warning")
        st.stop()
    left, top = view["window"][:2]
    if new_row is not None:
        is_defect = new_row["right_answer"] == "REJECT"
        real_boxes = [[box[0] - left, box[1] - top, box[2] - left, box[3] - top, box[4]] for box in new_row["real_boxes"]]
        answer, warning = f'<span class="mono">{photo}</span> {label_text(is_defect, new_row["defect_types"])}', None
    else:
        match = inference.find_match(cut["crop"], new_photo_files=tuple(new_photos["file"]))
        is_defect, real_boxes, warning = None, None, None
        answer = "Your photo is not one of the photos of this project, so the right answer is not known."
        if match is not None:
            if match["part"] == "new":
                known = new_photos.set_index("name").loc[match["name"]]
                is_defect, defect_types = known["right_answer"] == "REJECT", known["defect_types"]
                real_boxes = [[box[0] - left, box[1] - top, box[2] - left, box[3] - top, box[4]] for box in known["real_boxes"]]
            else:
                known = load_split().set_index("name").loc[match["name"]]
                is_defect, defect_types = bool(known["is_defect"]), known["defect_types"]
                if match["name"] in set(gallery["name"]):
                    real_boxes = gallery.set_index("name").loc[match["name"], "real_boxes"]
            answer = (f'Your photo is <span class="mono">{match["name"]}</span>, {PART_TEXT[match["part"]]}. '
                      f'It {label_text(is_defect, defect_types)}')
            if match["part"] == "train":
                learned = "YOLO learned" if is_defect else "Both models learned"
                warning = f"{learned} from this photo, so this check is not a fair test."
            elif match["part"] == "val":
                warning = "This photo helped to set the limits, so this check is not a fair test."

patchcore_score, yolo_score = view["patchcore_score"], view["yolo_score"]
patchcore_rejects, yolo_rejects = patchcore_score > limits["PatchCore"], yolo_score > limits["YOLO"]
if patchcore_rejects == yolo_rejects:
    sentence = f"Both models {'reject' if patchcore_rejects else 'pass'} this photo."
else:
    sentence = f"PatchCore {'rejects' if patchcore_rejects else 'passes'} this photo, YOLO {'rejects' if yolo_rejects else 'passes'} it."

st.html(f'<div class="verdict"><div class="say">{sentence}</div></div>')
with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", key="answer_row"):
    st.html(f'<div class="answer">{answer}</div>', width="content")
    show_truth = real_boxes is not None and st.toggle("Show real defects", value=True, key="truth")
if warning:
    notice(warning, kind="warning")

truth_html = real_box_html(real_boxes) if show_truth else ""
heat_figure = f"""<div class="frame"><img src="{view['photo_uri']}" alt="photo">
<img src="{view['heat_uri']}" alt="PatchCore heatmap">{truth_html}</div>"""
yolo_figure = f"""<div class="frame"><img src="{view['photo_uri']}" alt="photo">{truth_html}
{yolo_box_html(view['yolo_boxes'], limit=limits['YOLO'], chips='all')}</div>"""

if view["live"]:
    times = {name: f"{view['times'][name]:.0f} ms on {view['device']}" for name in ["PatchCore", "YOLO"]}
else:
    times = {name: f"{TIME_MS[name]['GPU']:.1f} ms per photo" for name in ["PatchCore", "YOLO"]}
patchcore_foot = f"""<div class="legend"><div class="item wide"><span class="glow"></span>unusual areas, full red at the limit</div></div>
<span class="time">{times['PatchCore']}</span>"""
shown_boxes = [box for box in view["yolo_boxes"] if box[4] > limits["YOLO"]]
if shown_boxes:
    count = "a box" if len(shown_boxes) == 1 else f"{len(shown_boxes)} boxes"
    items = f'<div class="item wide"><span class="boxkey"></span>{count} above the limit, with confidence</div>'
else:
    items = f'<div class="item wide"><span>No box above the limit. Strongest guess: <b>{yolo_score:.2g}</b></span></div>'
yolo_foot = f"""<div class="legend">{items}</div><span class="time">{times['YOLO']}</span>"""

st.space("small")
with st.container(key="panels"):
    left_panel, right_panel = st.columns(2, gap="medium")
with left_panel:
    st.html(panel("PatchCore", "Learned from good photos only", patchcore_score, limits["PatchCore"], 1.6, 3.8, False,
                  heat_figure, patchcore_foot, is_defect))
with right_panel:
    st.html(panel("YOLO", f"Learned from {counts['labeled']} photos with marked defects", yolo_score, limits["YOLO"], 1e-4, 1.0, True,
                  yolo_figure, yolo_foot, is_defect))

if view["live"]:
    photo_width, photo_height = view["photo_size"]
    footnote = (f"Checked live just now by the models of split {SPLIT} (YOLO with k = {K}), with the limits of notebook 5.1. "
                f"The gum was cut out of the {photo_width} × {photo_height} photo first, as in notebook 3.1. "
                f"Times are for this photo on the {view['device']} of the computer that runs the app"
                + (", measured after one untimed check that wakes the GPU up, like the warm-up of notebook 5.3." if view["device"] == "GPU" else "."))
else:
    footnote = (f"Saved results of split {SPLIT} (notebooks 4.1 and 4.3, YOLO with k = {K}) with the limits of notebook 5.1. "
                "Times are medians on the laptop GPU (notebook 5.3).")
st.html(f'<div class="footnote">{footnote}</div>')
