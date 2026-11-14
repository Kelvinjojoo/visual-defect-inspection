import streamlit as st

import common

st.set_page_config(page_title="GumCheck", page_icon=common.ICON_FILE, layout="wide", initial_sidebar_state="auto")
st.logo(common.LOGO_FILE, size="large", icon_image=common.ICON_FILE)
st.html(common.CSS)

INSPECT_PAGE = st.Page("views/inspect.py", title="Inspect a photo", icon=":material/center_focus_weak:", url_path="inspect", default=True)
RESULTS_PAGE = st.Page("views/results.py", title="Results", icon=":material/bar_chart:", url_path="results")
HOW_PAGE = st.Page("views/how_it_works.py", title="How it works", icon=":material/info:", url_path="how-it-works")
page = st.navigation([INSPECT_PAGE, RESULTS_PAGE, HOW_PAGE], position="hidden")
ACTIVE_HREF = {"Inspect a photo": "", "Results": "results", "How it works": "how-it-works"}[page.title]
st.html(f"""<style>
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{ACTIVE_HREF}"] {{ background: var(--surface); box-shadow: inset 0 0 0 1px var(--line); }}
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{ACTIVE_HREF}"] p,
[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{ACTIVE_HREF}"] span[data-testid="stIconMaterial"] {{ color: var(--ink); font-weight: 550; }}
</style>""")

with st.sidebar:
    st.page_link(INSPECT_PAGE, label="Inspect a photo", icon=":material/center_focus_weak:", width="stretch")
    st.page_link(RESULTS_PAGE, label="Results", icon=":material/bar_chart:", width="stretch")
    st.page_link(HOW_PAGE, label="How it works", icon=":material/info:", width="stretch")

page.run()
