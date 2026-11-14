import base64
import io
import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import ndimage

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(APP_DIR, "data")
GALLERY_DIR = os.path.join(DATA_DIR, "gallery")
STATIC_DIR = os.path.join(APP_DIR, "static")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TEST_PHOTO_DIR = os.path.join(ROOT_DIR, "test_photos")
SPLITS_DIR = os.path.join(ROOT_DIR, "splits")
BANK_FILE = os.path.join(ROOT_DIR, "models", "patchcore", "split_1_memory_bank.npz")
YOLO_FILE = os.path.join(ROOT_DIR, "models", "yolo", "split_1_k40", "weights", "last.pt")
LOGO_FILE = os.path.join(STATIC_DIR, "logo.svg")
ICON_FILE = os.path.join(STATIC_DIR, "icon.svg")

SPLIT = 1
K = 40
WIDTH, HEIGHT = 928, 608
HEAT_STRENGTH = 0.75
HEAT_BLUR = 8
FIND_SCALE = 4
GUM_BRIGHTNESS = 120
MODEL_COLORS = {"PatchCore": "#EB6834", "YOLO": "#2A78D6"}
TIME_MS = {"PatchCore": {"GPU": 49.3, "CPU": 153.1}, "YOLO": {"GPU": 30.5, "CPU": 105.8}}
POINTING = {"PatchCore": 97.9, "YOLO": 98.6}

CSS = """
<style>
@font-face { font-family: "Geist"; src: url("/app/static/fonts/Geist-Variable.woff2") format("woff2"); font-weight: 100 900; font-display: swap; }
@font-face { font-family: "Geist Mono"; src: url("/app/static/fonts/GeistMono-Variable.woff2") format("woff2"); font-weight: 100 900; font-display: swap; }
:root { --canvas: #F1F1EE; --side: #F7F7F5; --surface: #FFFFFF; --line: #E3E2DE; --grid: #ECEBE7; --ink: #161616; --muted: #74736F;
        --reject: #C8372D; --reject-bg: #FBEAE8; --reject-zone: #F4D3CF; --pass: #2E7D4F; --pass-bg: #E6F1E9;
        --patchcore: #EB6834; --yolo: #2A78D6; --note-bg: #FFF6E5; --note-ink: #8A5A00; }
html, body, .stApp, .stApp p, .stApp label, .stApp button, .stApp input, .stApp td, .stApp th { font-family: "Geist", sans-serif; }
.stApp { background: var(--canvas); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; pointer-events: none; }
[data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stToolbarActions"], [data-testid="stStatusWidget"], [data-testid="stDecoration"] { display: none; }
[data-testid="stExpandSidebarButton"] { pointer-events: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; width: 36px; height: 36px; }
[data-testid="stHeaderLogo"] { pointer-events: auto; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1280px; }
[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--line); }
[data-testid="stSidebarHeader"] { padding: 1.5rem 0.75rem 0.5rem 12px; }
[data-testid="stSidebarLogo"] { height: 30px !important; max-width: 100%; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0.6rem; }
.stMarkdown p { margin: 0; }

[data-testid="stSidebar"] [data-testid="stPageLink"] a { padding: 9px 12px; border-radius: 12px; gap: 12px; }
[data-testid="stSidebar"] [data-testid="stPageLink"] a span[data-testid="stIconMaterial"] { color: var(--muted); font-size: 20px; }
[data-testid="stSidebar"] [data-testid="stPageLink"] a p { font-size: 15px; color: var(--muted); }
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { background: rgba(22,22,22,0.04); }
.side-card { margin-top: 26px; background: var(--surface); border: 1px solid var(--line); border-radius: 16px; padding: 16px 16px 6px; }
.side-card h4 { font-size: 13px; font-weight: 600; margin: 0 0 12px; color: var(--ink); }
.side-card .row { margin-bottom: 12px; font-size: 13px; line-height: 1.45; color: var(--muted); }
.side-card .row b { display: block; color: var(--ink); font-weight: 600; font-size: 13.5px; }

.title, .stMarkdown h1.title { font-size: 34px; font-weight: 650; letter-spacing: -0.03em; line-height: 1.1; margin: 0 0 8px; padding: 0; color: var(--ink); }
.lede { font-size: 15.5px; color: var(--muted); max-width: 62ch; line-height: 1.5; margin: 0; }
.section-title { font-size: 20px; font-weight: 650; letter-spacing: -0.02em; color: var(--ink); margin: 0 0 4px; }
.section-sub { font-size: 14px; color: var(--muted); margin: 0 0 14px; line-height: 1.5; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 20px; padding: 20px 22px; }
.card-title { font-size: 16px; font-weight: 600; color: var(--ink); }
.card-sub { font-size: 13.5px; color: var(--muted); margin-top: 2px; line-height: 1.45; }
.mono { font-family: "Geist Mono", monospace; }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: none; }
.footnote { font-size: 12.5px; color: var(--muted); margin-top: 26px; line-height: 1.5; }

.st-key-gallery { background: var(--surface); border: 1px solid var(--line); border-radius: 20px; padding: 18px 18px 10px; gap: 0.6rem; }
.st-key-strip { gap: 6px; overflow-x: auto; scrollbar-width: thin; padding-bottom: 4px; }
[class*="st-key-thumb_"] { flex: 0 0 auto; position: relative; border-radius: 13px; padding: 5px; border: 1px solid transparent; }
[class*="st-key-thumb_"]:hover { border-color: var(--line); }
[class*="st-key-thumb_"][class*="_on"] { border-color: var(--ink); box-shadow: inset 0 0 0 1px var(--ink); }
[class*="st-key-pick_"] { position: absolute !important; inset: 0; z-index: 5; }
[class*="st-key-pick_"] button { width: 100%; height: 100%; opacity: 0; cursor: pointer; }
[class*="st-key-pick_"] button:focus-visible { opacity: 1; background: transparent; outline: 2px solid var(--ink); }
.thumb img { width: 100%; aspect-ratio: 928 / 608; object-fit: cover; border-radius: 9px; display: block; }
.thumb .name { font-family: "Geist Mono", monospace; font-size: 12px; color: var(--ink); margin-top: 7px; }
.thumb .tag { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; margin-top: 1px; white-space: nowrap; overflow: hidden; }
.thumb .tag span.text { overflow: hidden; text-overflow: ellipsis; }
[data-variant="segmented_control"] { min-height: 36px; padding: 4px 15px; background: var(--surface); }
[data-variant="segmented_control"] p { font-size: 13.5px; }
[data-variant="segmented_control"][aria-checked="true"] { background: var(--ink) !important; border-color: var(--ink) !important; }
[data-variant="segmented_control"][aria-checked="true"] p { color: #fff !important; }
[data-testid="stPopover"] button { background: var(--ink); color: #fff; border: none; border-radius: 12px; min-height: 44px; font-weight: 550; }
[data-testid="stPopover"] button:hover { background: #2b2b2b; color: #fff; }
[data-testid="stPopoverBody"] button { background: var(--surface); color: var(--ink); border: 1px solid var(--line); min-height: 0; }

.verdict .say { font-size: 26px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.2; color: var(--ink); padding: 6px 2px 0; }
.st-key-answer_row { gap: 10px 28px; margin-top: -6px; }
.answer { font-size: 15px; color: var(--muted); line-height: 1.5; }
.answer .mono { color: var(--ink); font-size: 14px; }
.answer b.reject { color: var(--reject); font-weight: 600; }
.answer b.pass { color: var(--pass); font-weight: 600; }
.st-key-truth p { font-size: 14px; color: var(--ink); white-space: nowrap; }
.notice { display: flex; gap: 10px; align-items: flex-start; background: var(--note-bg); color: var(--note-ink); border-radius: 14px; padding: 12px 16px; font-size: 14px; line-height: 1.5; }
.notice.info { background: #EEF3FB; color: #24508F; }
.notice .material { font-family: "Material Symbols Rounded"; font-size: 20px; line-height: 1.1; }

.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 22px; padding: 22px 22px 18px; height: 100%; box-sizing: border-box; display: flex; flex-direction: column; }
.st-key-panels [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-panels [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { height: 100%; }
.st-key-panels [data-testid="stColumn"] [data-testid="stElementContainer"] { flex: 1 1 auto; }
.st-key-panels [data-testid="stHtml"] { height: 100%; }
.panel-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.panel-name { font-size: 22px; font-weight: 650; letter-spacing: -0.02em; color: var(--ink); line-height: 1.2; }
.panel-sub { font-size: 13.5px; color: var(--muted); margin-top: 3px; }
.pill { font-family: "Geist Mono", monospace; font-weight: 600; font-size: 13px; letter-spacing: 0.05em; padding: 7px 13px; border-radius: 999px; white-space: nowrap; }
.pill.reject { color: var(--reject); background: var(--reject-bg); }
.pill.pass { color: var(--pass); background: var(--pass-bg); }
.score-row { display: flex; align-items: baseline; gap: 14px; margin-top: 18px; }
.score { font-family: "Geist Mono", monospace; font-size: 30px; font-weight: 500; letter-spacing: -0.02em; color: var(--ink); }
.score-label { font-size: 13px; color: var(--muted); }
.gauge { display: block; width: 100%; height: auto; margin: 6px 0 0; }
.zones { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--muted); margin: 2px 4px 14px; }
.zones .reject { color: var(--reject); }
.decision { display: flex; flex-direction: column; align-items: flex-end; gap: 7px; }
.mark { display: inline-flex; align-items: center; gap: 4px; font-size: 12.5px; }
.mark img { width: 15px; height: 15px; }
.mark.right { color: var(--pass); }
.mark.wrong { color: var(--reject); }
.frame { position: relative; border-radius: 14px; overflow: hidden; aspect-ratio: 928 / 608; background: #1b1b1b; }
.frame > img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.truth-box { position: absolute; border: 1.5px dashed #fff; border-radius: 4px; filter: drop-shadow(0 0 1px rgba(0,0,0,0.85)); }
.ybox { position: absolute; border: 2px solid var(--c); border-radius: 4px; }
.ybox span { position: absolute; left: -2px; bottom: 100%; margin-bottom: 3px; white-space: nowrap; background: var(--c); color: #fff; font-family: "Geist Mono", monospace; font-size: 11.5px; font-weight: 500; padding: 1px 6px; border-radius: 6px; }
.ybox.inside span { bottom: auto; top: 0; margin: 3px 0 0 3px; }
.panel-foot { display: flex; justify-content: space-between; align-items: flex-start; margin-top: 14px; font-size: 13px; color: var(--muted); gap: 16px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; }
.legend .item { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap; }
.legend .item.wide { white-space: normal; }
.legend .item b { font-family: "Geist Mono", monospace; font-weight: 500; color: var(--ink); }
.boxkey { width: 14px; height: 10px; border: 2px solid var(--yolo); border-radius: 3px; display: inline-block; flex: none; }
.glow { width: 46px; height: 8px; border-radius: 4px; background: linear-gradient(90deg, rgba(255,160,40,0), rgba(255,150,40,.55), #E0402F); display: inline-block; flex: none; }
.time { font-family: "Geist Mono", monospace; color: var(--ink); white-space: nowrap; flex: none; }

.page { display: flex; flex-direction: column; gap: 18px; }
.answer-card { background: var(--ink); color: #fff; border-radius: 22px; padding: 26px 28px; }
.answer-card .q { font-size: 14px; color: #B9B8B3; }
.answer-card .a { font-size: 24px; font-weight: 550; letter-spacing: -0.02em; line-height: 1.3; margin-top: 8px; max-width: 46ch; }
.answer-card .facts { display: flex; flex-wrap: wrap; gap: 14px 48px; margin-top: 22px; padding-top: 18px; border-top: 1px solid #2E2E2D; }
.answer-card .fact { font-size: 13.5px; color: #B9B8B3; line-height: 1.5; }
.answer-card .fact .who { display: flex; align-items: center; gap: 8px; color: #fff; font-weight: 550; }
.answer-card .fact .big { display: block; color: #fff; font-size: 24px; font-weight: 550; letter-spacing: -0.01em; margin: 4px 0 1px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.chart-card .key { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); margin: 12px 0 4px; }
.chart-card .key span { display: inline-flex; align-items: center; gap: 7px; }
.key .line { width: 18px; height: 2px; border-radius: 2px; display: inline-block; }
.chart { display: block; width: 100%; height: auto; overflow: visible; }
.chart.narrow { display: none; }
.chart text { font-family: "Geist", sans-serif; }
.chart .tick { font-size: 12px; fill: #74736F; font-variant-numeric: tabular-nums; }
.chart .lab { font-size: 13px; fill: #161616; font-weight: 550; }
.chart .lab-muted { font-size: 12.5px; fill: #74736F; }
.k-card { background: var(--surface); border: 1px solid var(--line); border-radius: 18px; padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; }
.k-card .k { font-size: 13px; color: var(--muted); }
.k-card .verdict-text { font-size: 16px; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); line-height: 1.3; min-height: 2.6em; }
.k-card .verdict-text.none { color: var(--muted); font-weight: 550; }
.k-card .splits { display: flex; flex-direction: column; gap: 7px; padding-top: 12px; border-top: 1px solid var(--grid); }
.k-card .splits div { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--ink); white-space: nowrap; }
.k-card .splits .s { color: var(--muted); width: 50px; flex: none; }
.k-card .splits .tie { color: var(--muted); }
.compare { width: 100%; border-collapse: collapse; font-size: 14px; margin: 0; }
.compare, .compare tr, .compare th, .compare td, .all, .all tr, .all th, .all td { border: 0 !important; background: transparent; }
.compare th { text-align: right; font-weight: 600; font-size: 13px; color: var(--ink); padding: 0 0 10px; }
.compare th:first-child { text-align: left; color: var(--muted); font-weight: 500; }
.compare th span { display: inline-flex; align-items: center; gap: 7px; }
.compare td { padding: 11px 0; border-top: 1px solid var(--grid) !important; text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.compare td:first-child { text-align: left; color: var(--ink); white-space: normal; padding-right: 12px; }
.compare td.better { color: var(--ink); font-weight: 600; }
.compare td + td { padding-left: 14px; }
.miss-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px; }
.miss .who { display: flex; align-items: center; gap: 7px; font-size: 13.5px; font-weight: 600; color: var(--ink); margin-top: 10px; }
.miss .why { font-size: 13px; color: var(--muted); margin-top: 3px; line-height: 1.45; }
.note { font-size: 13.5px; color: var(--muted); line-height: 1.5; margin-top: 14px; }
details.table-view { margin-top: 10px; }
details.table-view summary { cursor: pointer; font-size: 13.5px; color: var(--ink); font-weight: 550; list-style: none; display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
details.table-view summary::-webkit-details-marker { display: none; }
details.table-view[open] summary { margin-bottom: 12px; }
.all { width: 100%; font-size: 13.5px; background: var(--surface) !important; border: 1px solid var(--line) !important; border-collapse: separate; border-spacing: 0; border-radius: 14px; overflow: hidden; }
.all th, .all td { padding: 10px 14px; text-align: right; font-variant-numeric: tabular-nums; }
.all td { border-top: 1px solid var(--grid) !important; }
.all th { color: var(--muted); font-weight: 500; font-size: 12.5px; }
.all th:first-child, .all td:first-child { text-align: left; }

.steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.step { background: var(--surface); border: 1px solid var(--line); border-radius: 18px; padding: 14px; position: relative; display: flex; flex-direction: column; }
.step .what { margin-bottom: 12px; }
.step .num { width: 24px; height: 24px; border-radius: 50%; background: var(--ink); color: #fff; font-size: 12.5px; font-weight: 600; display: inline-flex; align-items: center; justify-content: center; }
.step .name { font-size: 15px; font-weight: 600; color: var(--ink); margin: 10px 0 2px; }
.step .what { font-size: 13px; color: var(--muted); line-height: 1.45; }
.step .pic { position: relative; border-radius: 10px; overflow: hidden; margin-top: auto; background: #1b1b1b; }
.step .pic img { display: block; width: 100%; height: 100%; object-fit: cover; }
.step .pic.photo { aspect-ratio: 928 / 608; display: flex; justify-content: center; }
.step .pic.photo .inner { position: relative; height: 100%; aspect-ratio: 1342 / 1118; }
.step .pic.crop { aspect-ratio: 928 / 608; }
.step .pic.white { background: #FAFAF8; border: 1px solid var(--grid); aspect-ratio: 928 / 608; display: flex; flex-direction: column; justify-content: center; padding: 10px 12px; gap: 6px; }
.window { position: absolute; border: 2px solid #fff; border-radius: 4px; box-shadow: 0 0 0 999px rgba(0,0,0,0.45); }
.step + .step::before { content: ""; position: absolute; left: -12px; top: 26px; width: 10px; height: 10px; border-top: 2px solid #B9B8B3; border-right: 2px solid #B9B8B3; transform: rotate(45deg); }
.patch-grid { position: absolute; inset: 0; background-image: linear-gradient(to right, rgba(255,255,255,0.32) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.32) 1px, transparent 1px); background-size: calc(100% / 58) calc(100% / 38); }
.ring { position: absolute; width: 30px; height: 30px; margin: -15px 0 0 -15px; border: 2px solid #fff; border-radius: 50%; box-shadow: 0 0 0 1px rgba(0,0,0,0.5); }
.model-card { display: flex; flex-direction: column; gap: 14px; }
.model-card .head { display: flex; align-items: center; gap: 10px; }
.model-card .head .name { font-size: 20px; font-weight: 650; letter-spacing: -0.02em; }
.model-card .head .tagline { font-size: 13.5px; color: var(--muted); }
.pair { display: grid; grid-template-columns: 1fr 22px 1fr; align-items: center; gap: 6px; }
.pair.caps { align-items: start; }
.pair .arrow { color: #B9B8B3; font-size: 18px; text-align: center; }
.pair .cap { font-size: 12.5px; color: var(--muted); line-height: 1.4; }
.howlist { display: flex; flex-direction: column; gap: 10px; margin: 0; padding: 0; list-style: none; }
.howlist li { display: grid; grid-template-columns: 24px 1fr; gap: 10px; font-size: 14px; line-height: 1.45; color: var(--ink); }
.howlist li .n { width: 22px; height: 22px; border-radius: 50%; border: 1px solid var(--line); font-size: 12px; color: var(--muted); display: inline-flex; align-items: center; justify-content: center; margin-top: 1px; }
.score-line { display: flex; align-items: center; gap: 10px; background: #F7F7F5; border-radius: 12px; padding: 10px 14px; font-size: 14px; color: var(--ink); }
.score-line .mono { font-size: 13px; }
.split-row { display: grid; grid-template-columns: 150px 1fr; align-items: center; gap: 14px; margin-top: 12px; }
.split-row .label { font-size: 13.5px; color: var(--ink); font-weight: 550; }
.split-row .label span { display: block; font-size: 12.5px; color: var(--muted); font-weight: 400; }
.seg { display: flex; gap: 2px; height: 28px; }
.seg div { border-radius: 4px; font-size: 12px; font-weight: 550; display: flex; align-items: center; padding: 0 8px; white-space: nowrap; overflow: hidden; }
.seg .train { background: #E4E3DE; color: var(--ink); }
.seg .val { background: #B5B4AE; color: var(--ink); }
.seg .test { background: var(--ink); color: #fff; }
.seg .out { background: repeating-linear-gradient(45deg, #E4E3DE 0 3px, #fff 3px 6px); }
.split-key { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 13px; color: var(--muted); margin-top: 16px; }
.split-key span { display: inline-flex; align-items: center; gap: 8px; }
.split-key i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.rules { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 18px; }
.rule { background: #F7F7F5; border-radius: 14px; padding: 14px; font-size: 13.5px; color: var(--muted); line-height: 1.45; }
.rule b { display: block; color: var(--ink); font-weight: 600; font-size: 14px; margin-bottom: 3px; }
.rule .material { font-family: "Material Symbols Rounded"; font-size: 20px; color: var(--ink); display: block; margin-bottom: 6px; }

@media (max-width: 900px) {
  .grid4, .steps { grid-template-columns: 1fr 1fr; }
  .step:nth-child(3)::before { display: none; }
}
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding-top: 4.2rem; }
  [data-variant="segmented_control"] { padding: 4px 11px; }
  .verdict .say { font-size: 22px; }
  .panel-foot { flex-direction: column; gap: 8px; }
  .grid2, .miss-grid, .rules { grid-template-columns: 1fr; }
  .answer-card { padding: 22px; }
  .answer-card .a { font-size: 20px; }
  .split-row { grid-template-columns: 1fr; gap: 6px; }
  .step::before { display: none; }
  .chart.wide { display: none; }
  .chart.narrow { display: block; }
}
</style>
"""


def find_gum_box(gray, scale, brightness):
    small = gray.resize((gray.width // scale, gray.height // scale), Image.BILINEAR)
    bright = ndimage.uniform_filter(np.asarray(small, dtype=float), 3) > brightness
    pieces, n_pieces = ndimage.label(bright)
    sizes = np.bincount(pieces.ravel())
    sizes[0] = 0
    rows, cols = np.nonzero(pieces == sizes.argmax())
    return cols.min() * scale, rows.min() * scale, (cols.max() + 1) * scale, (rows.max() + 1) * scale


def place_window(gum_box, window_size, photo_size):
    left, top, right, bottom = gum_box
    width, height = window_size
    photo_width, photo_height = photo_size
    window_left = (left + right) // 2 - width // 2
    window_top = (top + bottom) // 2 - height // 2
    window_left = min(max(window_left, 0), photo_width - width)
    window_top = min(max(window_top, 0), photo_height - height)
    return window_left, window_top, window_left + width, window_top + height


def open_photo(photo_bytes):
    return Image.open(io.BytesIO(photo_bytes)).convert("RGB")


def cut_gum(photo, window_size, scale, brightness):
    gum_box = find_gum_box(photo.convert("L"), scale=scale, brightness=brightness)
    window = place_window(gum_box, window_size, photo.size)
    return photo.crop(window), tuple(int(value) for value in window), tuple(int(value) for value in gum_box)


@st.cache_data(show_spinner=False)
def read_bytes(path):
    with open(path, "rb") as file:
        return file.read()


@st.cache_data(show_spinner=False, max_entries=32)
def cut_photo(photo_bytes):
    photo = open_photo(photo_bytes)
    crop, window, gum_box = cut_gum(photo, window_size=(WIDTH, HEIGHT), scale=FIND_SCALE, brightness=GUM_BRIGHTNESS)
    return {"crop": crop, "window": window, "gum_box": gum_box, "photo_size": photo.size}


def short_name(rel_path):
    return rel_path.split("Images/")[-1].rsplit(".", 1)[0]


def encode(image, fmt):
    buffer = io.BytesIO()
    if fmt == "JPEG":
        image.convert("RGB").save(buffer, format=fmt, quality=88)
    else:
        image.save(buffer, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


def svg_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def icon(paths, color):
    return svg_uri(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.7" '
                   f'stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')


CHECK = icon('<path d="M5 12.5l4.5 4.5L19 7.5"/>', "#2E7D4F")
CROSS = icon('<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>', "#C8372D")


def box_style(left, top, right, bottom):
    return (f"left:{100 * left / WIDTH:.2f}%;top:{100 * top / HEIGHT:.2f}%;"
            f"width:{100 * (right - left) / WIDTH:.2f}%;height:{100 * (bottom - top) / HEIGHT:.2f}%;")


def real_box_html(boxes):
    return "".join(f'<div class="truth-box" style="{box_style(*box[:4])}"></div>' for box in boxes)


def yolo_box_html(boxes, limit, chips):
    shown = [box for box in boxes if box[4] > limit]
    top_confidence = max((box[4] for box in shown), default=None)
    parts = []
    for left, top, right, bottom, confidence in shown:
        chip = f"<span>{confidence:.2f}</span>" if chips == "all" or confidence == top_confidence else ""
        inside = " inside" if top < 40 else ""
        parts.append(f'<div class="ybox{inside}" style="{box_style(left, top, right, bottom)}--c:{MODEL_COLORS["YOLO"]}">{chip}</div>')
    return "".join(parts)


def html_block(body):
    st.markdown(" ".join(line.strip() for line in body.strip().splitlines()), unsafe_allow_html=True)


@st.cache_data
def image_uri(path, size):
    image = Image.open(path)
    if size:
        image = image.resize(size, Image.LANCZOS)
    return encode(image, "JPEG")


@st.cache_data
def load_limits():
    limits = pd.read_csv(os.path.join(RESULTS_DIR, "evaluation", "score_limits.csv"))
    limits = limits[limits["split"] == SPLIT].set_index("model")["limit"]
    return {"PatchCore": float(limits["PatchCore"]), "YOLO": float(limits[f"YOLO k = {K}"])}


@st.cache_data
def load_patch_grids():
    with np.load(os.path.join(RESULTS_DIR, "patchcore", f"split_{SPLIT}_patch_scores.npz")) as saved:
        grids = dict(zip([short_name(str(image)) for image in saved["images"]], saved["grids"]))
    scores = pd.read_csv(os.path.join(RESULTS_DIR, "patchcore", f"split_{SPLIT}_scores.csv"))
    normal_val = scores.loc[(scores["part"] == "val") & ~scores["is_defect"], "image"].map(short_name)
    heat_low = float(np.median(np.stack([grids[name] for name in normal_val])))
    return grids, heat_low


@st.cache_data
def load_gallery():
    gallery = pd.read_csv(os.path.join(DATA_DIR, "gallery.csv")).fillna({"defect_types": ""})
    gallery["real_boxes"] = gallery["real_boxes"].map(json.loads)
    patchcore = pd.read_csv(os.path.join(RESULTS_DIR, "patchcore", f"split_{SPLIT}_scores.csv"))
    patchcore["name"] = patchcore["image"].map(short_name)
    patchcore = patchcore.set_index("name")
    yolo = pd.read_csv(os.path.join(RESULTS_DIR, "yolo", f"split_{SPLIT}_scores.csv"))
    yolo = yolo[yolo["k"] == K].copy()
    yolo["name"] = yolo["image"].map(short_name)
    yolo = yolo.set_index("name")
    boxes = pd.read_csv(os.path.join(RESULTS_DIR, "yolo", f"split_{SPLIT}_boxes.csv"))
    boxes = boxes[boxes["k"] == K]
    gallery["patchcore_score"] = gallery["name"].map(patchcore["score"])
    gallery["best_patch"] = [(int(patchcore.loc[name, "best_row"]), int(patchcore.loc[name, "best_col"])) for name in gallery["name"]]
    gallery["yolo_score"] = gallery["name"].map(yolo["score"])
    gallery["yolo_boxes"] = [boxes.loc[boxes["crop_image"] == crop, ["left", "top", "right", "bottom", "confidence"]].values.tolist()
                             for crop in gallery["crop_image"]]
    return gallery


@st.cache_data
def load_new_photos():
    answers = pd.read_csv(os.path.join(TEST_PHOTO_DIR, "answers.csv")).fillna({"defect_types": ""})
    answers["real_boxes"] = answers["real_boxes"].map(json.loads)
    answers["name"] = answers["file"].str.replace(".jpg", "", regex=False)
    return answers


@st.cache_data
def load_test_results():
    return pd.read_csv(os.path.join(RESULTS_DIR, "evaluation", "test_results.csv"))


@st.cache_data
def load_val_scores():
    scores = pd.read_csv(os.path.join(RESULTS_DIR, "patchcore", f"split_{SPLIT}_scores.csv"))
    val = scores[scores["part"] == "val"]
    return sorted(val.loc[~val["is_defect"], "score"].tolist()), sorted(val.loc[val["is_defect"], "score"].tolist())


@st.cache_data
def load_split():
    split = pd.read_csv(os.path.join(SPLITS_DIR, f"split_{SPLIT}.csv")).fillna({"defect_types": ""})
    split["name"] = split["image"].map(short_name)
    return split


def split_counts():
    split = load_split()
    good, defect = split[~split["is_defect"]], split[split["is_defect"]]
    return {"train good": int((good["split"] == "train").sum()), "val good": int((good["split"] == "val").sum()),
            "test good": int((good["split"] == "test").sum()), "left out": int((good["split"] == "excluded").sum()),
            "train defect": int((defect["split"] == "train").sum()), "val defect": int((defect["split"] == "val").sum()),
            "test defect": int((defect["split"] == "test").sum()), "labeled": int(defect[f"yolo_{K}"].sum())}


def heat_image(grid, heat_low, limit, strength, blur):
    heat = np.asarray(Image.fromarray(grid.astype(np.float32), mode="F").resize((WIDTH, HEIGHT), Image.BILINEAR))
    heat = ndimage.gaussian_filter(heat, blur)[::2, ::2]
    level = np.clip((heat - heat_low) / (limit - heat_low), 0, 1)
    orange, red = np.array([255, 160, 40]), np.array([224, 64, 47])
    rgb = (1 - level[..., None]) * orange + level[..., None] * red
    alpha = (level ** 2.2) * 255 * strength
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")


def heat_uri(grid, heat_low, limit, strength):
    return encode(heat_image(grid, heat_low=heat_low, limit=limit, strength=strength, blur=HEAT_BLUR), "PNG")


def pct(count, total):
    return 100 * count / total


@st.cache_data(show_spinner=False)
def saved_heat_uri(name, limit):
    grids, heat_low = load_patch_grids()
    return heat_uri(grids[name], heat_low=heat_low, limit=limit, strength=HEAT_STRENGTH)


@st.cache_data(show_spinner=False)
def bank_size():
    if not os.path.exists(BANK_FILE):
        return None
    with np.load(BANK_FILE) as bank:
        return int(bank["features"].shape[0])
