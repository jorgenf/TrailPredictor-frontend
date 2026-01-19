# dbc.py (Streamlit read-only, Supabase)
import streamlit as st
import pandas as pd
import json
from supabase import create_client, Client
from shapely.geometry import shape

from shapely.geometry import shape
import pandas as pd
import json
import util
from datetime import timezone, datetime, timedelta

# --- Supabase client ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# --- Fetch trail predictions ---
def fetch_trail_predictions(after):
    # Fetch trail_predictions
    batch_size=1000
    all_rows = []
    offset = 0

    while True:
        resp = (
            supabase
            .table("trail_predictions")
            .select("*")
            .range(offset, offset + batch_size - 1)
            .gte("timestamp", after)
            .execute()
        )

        data = resp.data
        if not data:
            break

        all_rows.extend(data)
        offset += batch_size

    df_tp = pd.DataFrame(all_rows)
    #resp_tp = supabase.table("trail_predictions").select("*").execute()
    #df_tp = pd.DataFrame(resp_tp.data)

    # Fetch trail_segments
    resp_ts = supabase.table("trail_segments").select("id, name, area_id").execute()
    df_ts = pd.DataFrame(resp_ts.data)

    # Merge like SQL join
    df = df_tp.merge(df_ts, left_on="trail_id", right_on="id", how="left")
    df.rename(columns={"name": "trail_name"}, inplace=True)
    df.drop(columns=["id"], inplace=True)  # drop redundant id

    # Apply util calculations
    
    df = util.calculate_damage_score(df)
    df = util.calculate_condition_score(df)
    df = util.calculate_median_speed_score(df)

    return df



# --- Fetch area predictions ---
def fetch_area_predictions(df_trails, after):
    """
    Fetches area predictions and joins areas for area_name.
    Applies calculate_area_scores using df_trails.
    """
    batch_size=1000
    all_rows = []
    offset = 0

    while True:
        resp = (
            supabase
            .table("area_predictions")
            .select("*, areas(name)")
            .range(offset, offset + batch_size - 1)
            .gte("timestamp", after)
            .execute()
        )

        data = resp.data
        if not data:
            break

        all_rows.extend(data)
        offset += batch_size

    df = pd.DataFrame(all_rows)

    #resp = supabase.table("area_predictions").select("*, areas(name)").execute()
    #df = pd.DataFrame(resp.data)

    # Flatten areas
    if "areas" in df.columns:
        df["area_name"] = df["areas"].apply(lambda x: x["name"] if x else None)
        df.drop(columns=["areas"], inplace=True)

    # Apply your utility calculations
    import util
    df = util.calculate_area_scores(df, df_trails)
    return df

# --- Fetch trail segments ---


from shapely.geometry import shape
import pandas as pd

def fetch_segments():
    """
    Fetch trail_segments from Supabase, return DataFrame compatible with old get_segments(),
    and parse geometry correctly for plotting.
    """
    # Fetch all columns from trail_segments
    resp = supabase.table("trail_segments").select("*").execute()
    if not resp.data:
        return pd.DataFrame(
            columns=["id", "name", "area_id", "length_m", "elevation_drop_m",
                     "difficulty", "coordinates", "geometry", "coords"]
        )

    df = pd.DataFrame(resp.data)

    # Parse geometry for plotting
    def parse_geom(g):
        if g is None:
            return None
        if isinstance(g, dict):
            # Supabase returns GeoJSON as dict
            return shape(g)
        elif isinstance(g, str):
            # Maybe WKB hex string (unlikely with Supabase)
            try:
                from shapely import wkb
                return wkb.loads(bytes.fromhex(g))
            except Exception:
                return None
        elif isinstance(g, (bytes, bytearray, memoryview)):
            try:
                from shapely import wkb
                if isinstance(g, memoryview):
                    g = g.tobytes()
                return wkb.loads(g)
            except Exception:
                return None
        return None

    df["geometry"] = df["coordinates"].apply(parse_geom)

    # Extract coordinates for plotting (lat, lon)
    df["coords"] = df["geometry"].apply(
        lambda geom: [(pt[1], pt[0]) for pt in geom.coords] if geom else []
    )

    return df


# --- Unified fetch function for app ---
def fetch_predictions():
    """
    Returns df_areas, df_trails exactly like your original app expected.
    """
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    df_trails = fetch_trail_predictions(three_days_ago)
    if df_trails is None:
        return None, None
    df_areas = fetch_area_predictions(df_trails, three_days_ago)
    if df_areas is None:
        return None, None
    return df_areas, df_trails
