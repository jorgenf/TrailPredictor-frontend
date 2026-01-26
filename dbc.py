# dbc.py (Streamlit + Supabase, cache-safe)

import streamlit as st
import pandas as pd
from supabase import create_client, Client
from shapely.geometry import shape
from shapely import wkb
from datetime import datetime, timedelta, timezone
import util

# --------------------------------------------------
# SUPABASE CLIENT
# --------------------------------------------------

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --------------------------------------------------
# RAW DATA FETCHES (CACHED, SERIALIZABLE ONLY)
# --------------------------------------------------

@st.cache_data(ttl=7200)
def fetch_trail_predictions_raw(after: str) -> pd.DataFrame:
    batch_size = 1000
    offset = 0
    all_rows = []

    while True:
        resp = (
            supabase
            .table("trail_predictions")
            .select("*")
            .gte("timestamp", after)
            .order("timestamp", desc=False)
            .range(offset, offset + batch_size - 1)
            .execute()
        )

        data = resp.data
        if not data:
            break

        all_rows.extend(data)
        offset += batch_size

    return pd.DataFrame(all_rows)


@st.cache_data(ttl=7200)
def fetch_area_predictions_raw(after: str) -> pd.DataFrame:
    batch_size = 1000
    offset = 0
    all_rows = []

    while True:
        resp = (
            supabase
            .table("area_predictions")
            .select("*")
            .gte("timestamp", after)
            .order("timestamp", desc=False)
            .range(offset, offset + batch_size - 1)
            .execute()
        )

        data = resp.data
        if not data:
            break

        all_rows.extend(data)
        offset += batch_size

    df = pd.DataFrame(all_rows)

    # Flatten areas relation if present
    if "areas" in df.columns:
        df["area_name"] = df["areas"].apply(lambda x: x["name"] if x else None)
        df.drop(columns=["areas"], inplace=True)

    return df


@st.cache_data(ttl=7200)
def fetch_segments_raw() -> pd.DataFrame:
    resp = supabase.table("trail_segments").select("*").execute()

    if not resp.data:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "area_id",
                "length_m",
                "elevation_drop_m",
                "difficulty",
                "coordinates",
            ]
        )

    return pd.DataFrame(resp.data)

# --------------------------------------------------
# PROCESSED DATA (CACHED, STILL NO GEOMETRY)
# --------------------------------------------------

@st.cache_data(ttl=7200)
def fetch_predictions_processed():
    """
    Returns df_areas, df_trails
    Fully processed but WITHOUT shapely geometry.
    """
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()

    # --- Trails ---
    df_tp = fetch_trail_predictions_raw(three_days_ago)
    df_ts = fetch_segments_raw()

    if df_tp.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_trails = df_tp.merge(
        df_ts[["id", "name", "area_id"]],
        left_on="trail_id",
        right_on="id",
        how="left"
    )

    df_trails.rename(columns={"name": "trail_name"}, inplace=True)
    df_trails.drop(columns=["id"], inplace=True)
    df_trails["trail_name"] = df_trails["trail_name"].fillna("Unnamed Trail")

    # Utility calculations
    df_trails = util.calculate_damage_score(df_trails)
    df_trails = util.calculate_condition_score(df_trails)
    df_trails = util.calculate_median_speed_score(df_trails)

    # --- Areas ---
    df_areas = fetch_area_predictions_raw(three_days_ago)
    if not df_areas.empty:
        df_areas = util.calculate_area_scores(df_areas, df_trails)

    return df_areas, df_trails

# --------------------------------------------------
# GEOMETRY PROCESSING (NOT CACHED)
# --------------------------------------------------

def fetch_segments() -> pd.DataFrame:
    """
    Fetch trail segments and parse geometry for mapping.
    NOT cached because shapely objects are not serializable.
    """
    df = fetch_segments_raw()

    def parse_geom(g):
        if g is None:
            return None
        if isinstance(g, dict):
            return shape(g)
        if isinstance(g, str):
            try:
                return wkb.loads(bytes.fromhex(g))
            except Exception:
                return None
        if isinstance(g, (bytes, bytearray, memoryview)):
            try:
                if isinstance(g, memoryview):
                    g = g.tobytes()
                return wkb.loads(g)
            except Exception:
                return None
        return None

    df["geometry"] = df["coordinates"].apply(parse_geom)
    df["coords"] = df["geometry"].apply(
        lambda g: [(pt[1], pt[0]) for pt in g.coords] if g else []
    )

    return df

# --------------------------------------------------
# PUBLIC API FOR APP.PY
# --------------------------------------------------

def fetch_predictions():
    """
    App-facing function.
    """
    return fetch_predictions_processed()
