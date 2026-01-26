# dbc.py (Streamlit + Supabase)
import streamlit as st
import pandas as pd
from supabase import create_client, Client
from shapely.geometry import shape
from shapely import wkb
import util
from datetime import datetime, timedelta, timezone

# --- Supabase client ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# ---------------------------
# CACHED RAW DATA FETCHES
# ---------------------------

@st.cache_data(ttl=7200)
def fetch_trail_predictions_raw(after: str) -> pd.DataFrame:
    """
    Fetch all rows from trail_predictions after a given timestamp.
    """
    batch_size = 1000
    all_rows = []
    offset = 0

    while True:
        resp = (
            supabase.table("trail_predictions")
            .select("*")  # keep all columns
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
    """
    Fetch area_predictions after a given timestamp, keep all columns.
    """
    batch_size = 1000
    all_rows = []
    offset = 0

    while True:
        resp = (
            supabase.table("area_predictions")
            .select("*")  # keep all columns
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

    # Flatten areas column if present
    if "areas" in df.columns:
        df["area_name"] = df["areas"].apply(lambda x: x["name"] if x else None)
        df.drop(columns=["areas"], inplace=True)

    return df


@st.cache_data(ttl=7200)
def fetch_segments_raw() -> pd.DataFrame:
    """
    Fetch trail_segments, raw data only.
    """
    resp = supabase.table("trail_segments").select("*").execute()
    if not resp.data:
        return pd.DataFrame(
            columns=[
                "id", "name", "area_id", "length_m", "elevation_drop_m",
                "difficulty", "coordinates", "geometry", "coords"
            ]
        )
    return pd.DataFrame(resp.data)


# ---------------------------
# PROCESSING FUNCTIONS (NOT CACHED)
# ---------------------------

def fetch_trail_predictions(after: str) -> pd.DataFrame:
    df_tp = fetch_trail_predictions_raw(after)
    df_ts = fetch_segments_raw()

    if df_tp.empty or df_ts.empty:
        return pd.DataFrame()

    # Merge trail predictions with segments
    df = df_tp.merge(df_ts[["id", "name", "area_id"]], left_on="trail_id", right_on="id", how="left")
    df.rename(columns={"name": "trail_name"}, inplace=True)
    df.drop(columns=["id"], inplace=True)
    df["trail_name"] = df["trail_name"].fillna("Unnamed Trail")

    # Apply util calculations
    df = util.calculate_damage_score(df)
    df = util.calculate_condition_score(df)
    df = util.calculate_median_speed_score(df)

    return df


def fetch_area_predictions(df_trails: pd.DataFrame, after: str) -> pd.DataFrame:
    df_areas = fetch_area_predictions_raw(after)
    if df_areas.empty or df_trails.empty:
        return pd.DataFrame()

    # Apply util calculations
    df_areas = util.calculate_area_scores(df_areas, df_trails)
    return df_areas


def fetch_segments() -> pd.DataFrame:
    """
    Fetch trail_segments and parse geometries for mapping.
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
            if isinstance(g, memoryview):
                g = g.tobytes()
            try:
                return wkb.loads(g)
            except Exception:
                return None
        return None

    df["geometry"] = df["coordinates"].apply(parse_geom)
    df["coords"] = df["geometry"].apply(
        lambda g: [(pt[1], pt[0]) for pt in g.coords] if g else []
    )
    return df


def fetch_predictions():
    """
    Unified fetch function for the app.
    """
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    df_trails = fetch_trail_predictions(three_days_ago)
    df_areas = fetch_area_predictions(df_trails, three_days_ago)
    return df_areas, df_trails
