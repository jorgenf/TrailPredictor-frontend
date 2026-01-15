# dbc.py (Streamlit read-only, Supabase)
import streamlit as st
import pandas as pd
import json
from supabase import create_client, Client
from shapely.geometry import shape

from shapely import wkb
from shapely.geometry import shape
import pandas as pd
import json

# --- Supabase client ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)



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
    df_trails = fetch_trail_predictions()
    if df_trails is None:
        return None, None
    df_areas = fetch_area_predictions(df_trails)
    if df_areas is None:
        return None, None
    return df_areas, df_trails
