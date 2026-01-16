from pathlib import Path
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
from streamlit_autorefresh import st_autorefresh
import folium
from streamlit_folium import st_folium
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import math
from shapely.geometry import shape, mapping
import numpy as np
from shapely.geometry import LineString, Point
from shapely import wkb
import util
from supabase import create_client, Client
import streamlit as st
import dbc
#st_autorefresh(interval=5000, key="datarefresh")



# --------------------------------------------------
# CONFIG
# --------------------------------------------------



st.set_page_config(
    page_title="MTB Trail Conditions",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bungee&display=swap');

.crazy-title {
    font-family: 'Bungee', cursive;
    font-size: 3.5rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: 0.2em;
}
</style>
""", unsafe_allow_html=True)


now = pd.Timestamp(datetime.now()).floor('h').strftime("%Y-%m-%d %H:%M")

BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "assets" / "transparent_logo.png"


col1, col2 = st.columns([1, 8])

with col1:
    st.image(LOGO_PATH, width=120)

with col2:
    st.markdown(
    '<div class="crazy-title">Rideable</div>',
    unsafe_allow_html=True
)

st.caption("Model-based predictions using terrain, weather & surface indicators")


# --------------------------------------------------
# DATA FETCHING
# --------------------------------------------------


@st.cache_data(ttl=600)
def fetch_predictions():
    """
    Returns df_areas, df_trails with all util calculations applied.
    """
    df_areas, df_trails = dbc.fetch_predictions()
    return df_areas, df_trails

@st.cache_data(ttl=600)
def fetch_segments():
    # Fetch segments from your database / Supabase
    df = dbc.fetch_segments()  # This returns the same as get_segments()

    def parse_geom(g):
        if g is None:
            return None
        elif isinstance(g, dict):
            # Supabase returns GeoJSON as dict
            return shape(g)
        elif isinstance(g, str):
            # WKB hex string (legacy)
            try:
                return wkb.loads(bytes.fromhex(g))
            except Exception:
                return None
        elif isinstance(g, (bytes, bytearray, memoryview)):
            # Convert memoryview to bytes
            if isinstance(g, memoryview):
                g = g.tobytes()
            try:
                return wkb.loads(g)
            except Exception:
                return None
        else:
            # Unknown type
            return None

    # Parse geometry
    df['geometry'] = df['coordinates'].apply(parse_geom)

    # Extract coordinates for plotting
    df['coords'] = df['geometry'].apply(
        lambda geom: [(pt[1], pt[0]) for pt in geom.coords] if geom else []
    )

    return df



df_areas, df_trails = fetch_predictions()
df_segments = fetch_segments()

if 'trail_lines' not in st.session_state:
    st.session_state.trail_lines = {}
    for _, segment in df_segments.iterrows():
        trail_name = segment['name']
        st.session_state.trail_lines[trail_name] = segment['geometry']

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def latest_timestamp(trail_dict):
    return sorted(
        trail_dict.keys(),
        key=lambda x: datetime.fromisoformat(x),
    )[-1]


from shapely.geometry import LineString, mapping

def smooth_and_curve_feature(feature, tolerance=0.0001, densify_factor=5):
    geom = feature["geometry"]

    if geom["type"] != "LineString":
        return feature

    # 1️⃣ Simplify slightly
    line = LineString(geom["coordinates"])
    simplified = line.simplify(tolerance, preserve_topology=True)

    # 2️⃣ Interpolate points
    curved_coords = densify_coords(
        list(simplified.coords),
        factor=densify_factor
    )

    feature["geometry"]["coordinates"] = curved_coords
    return feature

def densify_coords(coords, factor=5):
    """
    Adds interpolated points between coordinates.
    Works with [lon, lat] or [lon, lat, z]
    """
    new_coords = []

    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]

        lon1, lat1 = p1[0], p1[1]
        lon2, lat2 = p2[0], p2[1]

        z1 = p1[2] if len(p1) > 2 else None
        z2 = p2[2] if len(p2) > 2 else None

        for t in np.linspace(0, 1, factor, endpoint=False):
            lon = lon1 + (lon2 - lon1) * t
            lat = lat1 + (lat2 - lat1) * t

            if z1 is not None and z2 is not None:
                z = z1 + (z2 - z1) * t
                new_coords.append([lon, lat, z])
            else:
                new_coords.append([lon, lat])

    new_coords.append(coords[-1])
    return new_coords



# --------------------------------------------------
# BUILD TRAIL TABLE 
# --------------------------------------------------
def get_direction(sin, cos):
    # Step 1: Calculate the angle in radians
    angle_rad = math.atan2(sin, cos)

    # Step 2: Convert angle to degrees (0 degrees = North, 90 degrees = East, etc.)
    angle_deg = math.degrees(angle_rad)
    # Adjust angle to fit compass convention (0° = North, increasing clockwise)
    angle_deg = (angle_deg + 360) % 360  # This makes sure the angle is always positive

    # Step 3: Map the angle to a compass direction
    compass_directions = [
    "N ↑", "NNE ↗", "NE →", "ENE ↗", "E →", "ESE ↘", "SE ↓", "SSE ↘", 
    "S ↓", "SSW ↙", "SW ←", "WSW ↖", "W ←", "WNW ↖", "NW →", "NNW ↗"
    ]

    # Divide the 360 degrees into 16 parts, each representing a compass direction
    direction_index = int((angle_deg + 11.25) // 22.5)
    if direction_index == 16:
        direction_index = 0
    # Step 4: Get the direction
    compass_direction = compass_directions[direction_index]
    return compass_direction


# --------------------------------------------------
# MAP
# --------------------------------------------------
if "selected_trail" not in st.session_state:
    st.session_state.selected_trail = None

# Ensure Time is datetime and rounded to the hour
df_trails['timestamp'] = pd.to_datetime(df_trails['timestamp'], errors='coerce')
#df_trails['timestamp'] = df_trails['timestamp'].astype('datetime64[ns]')
df_trails['time_hour'] = df_trails['timestamp'].dt.floor('h')

# Hourly slider
min_time = df_trails['time_hour'].min()
max_time = df_trails['time_hour'].max()

selected_time = st.slider(
    "Select date & hour",
    min_value=min_time.to_pydatetime(),
    max_value=max_time.to_pydatetime(),
    value=min_time.to_pydatetime(),
    step=timedelta(hours=1),
    format="YYYY-MM-DD HH:mm",
    key="time_slider"
)
if "last_slider_value" not in st.session_state:
    st.session_state.last_slider_value = selected_time

# Filter for selected hour
df_hour = df_trails[df_trails['time_hour'] == pd.Timestamp(selected_time)]
condition_map = dict(zip(df_hour['trail_name'], df_hour['condition_score']))

# Map colors
def condition_color(condition_value):
    if condition_value == -1:
        return "grey"
    condition_value = max(0, min(10, condition_value))
    if condition_value <= 5:
        r = 255
        g = int(255 * (condition_value / 5))
        b = 0
    else:
        r = int(255 * ((10 - condition_value) / 5))
        g = 255
        b = 0
    return f'#{r:02x}{g:02x}{b:02x}'

difficulty_color = {
    "green": "#4E944F",
    "blue": "#3498db",
    "red": "#D9534F",
    "black": "#2C2C2C"
}
HIGHLIGHT_COLOR = "#00FFFF"   # cyan
HIGHLIGHT_WEIGHT = 7
NORMAL_WEIGHT = 4


# Keep map center & zoom across rerenders
if 'last_center' not in st.session_state:
    st.session_state.last_center = [61.1689, 7.2786]
    st.session_state.last_zoom = 12

# Create base map
#m = folium.Map(
#    location=st.session_state.last_center,
#    zoom_start=st.session_state.last_zoom,
#    tiles="https://tiles-eu.stadiamaps.com/tiles/alidade_satellite/{z}/{x}/{y}{r}.jpg",
#    control_scale=True,
#    attr="© CNES, Distribution Airbus DS, © Airbus DS, © PlanetObserver (Contains Copernicus Data)"
#)

m = folium.Map(
    location=st.session_state.last_center,
    zoom_start=st.session_state.last_zoom,
    tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',  # OSM tile URL
    control_scale=True,
    attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
)

# Disable blue focus box around features
m.get_root().html.add_child(folium.Element("""
<style>
.leaflet-interactive:focus {
    outline: none;
}
</style>
"""))

# Draw trails
for idx, row in df_segments.iterrows():
    trail_name = row['name']
    trail_points = row['coords']
    if not trail_points:
        continue

    trailhead = trail_points[0]
    trail_diff = row['difficulty']
    trail_condition = condition_map.get(trail_name, -1) 

    # Is this trail selected in Streamlit?
    is_selected = trail_name == st.session_state.selected_trail

    # Polyline
    folium.PolyLine(
        locations=trail_points,
        color=HIGHLIGHT_COLOR if is_selected else condition_color(trail_condition),
        weight=HIGHLIGHT_WEIGHT if is_selected else NORMAL_WEIGHT,
        opacity=1.0 if is_selected else 0.8,
        tooltip=folium.Tooltip(
            f"Trail: {trail_name}<br>"
            f"Difficulty: {trail_diff}<br>"
            f"Length: {row.get('length_m', 'N/A')}<br>"
            f"Altitude drop: {row.get('elevation_drop_m', 'N/A')}", 
            sticky=True
        )
    ).add_to(m)

    # Trailhead marker
    folium.CircleMarker(
        location=trailhead,
        radius=3,
        color=difficulty_color.get(trail_diff, 'gray'),
        fill=True,
        fill_color=difficulty_color.get(trail_diff, 'gray'),
        fill_opacity=1,
        tooltip=folium.Tooltip(f"{trail_name} - Difficulty {trail_diff}", sticky=True)
    ).add_to(m)

# Difficulty legend
legend = folium.Element("""
<div style="
    position: fixed;
    bottom: 25px;
    right: 20px;
    width: 200px;
    padding: 10px;
    background: white;
    color: #111;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    font-size: 14px;
    z-index: 9999;
">
<b>Trail Difficulty</b><br>
<span style="color:#4E944F;">●</span> Easy<br>
<span style="color:#3498db;">●</span> Medium<br>
<span style="color:#D9534F;">●</span> Hard<br>
<span style="color:#2C2C2C;">●</span> Expert
</div>
""")
m.get_root().html.add_child(legend)

output_map = st_folium(m, height=600, width="100%")

clicked = output_map.get("last_clicked", None)
clicked_trail_name = None

if clicked:
    clicked_lat = clicked["lat"]
    clicked_lon = clicked["lng"]
    click_point = Point(clicked_lon, clicked_lat)
    for trail_name, line in st.session_state.trail_lines.items():
        if line.buffer(0.0005).contains(click_point):
            st.session_state.selected_trail = trail_name
            break

    if output_map:
        if "center" in output_map:
            center = output_map["center"]
            st.session_state.last_center = [center["lat"], center["lng"]]
        if "zoom" in output_map:
            st.session_state.last_zoom = output_map["zoom"]

    # Update last slider value (for other logic if needed)
    st.session_state.last_slider_value = selected_time
    if st.session_state.selected_trail:
        st.rerun()


# --------------------------------------------------
# TRAIL DETAIL VIEW
# --------------------------------------------------
st.subheader("Trail details")

if st.session_state.selected_trail is not None:
    selected_trail = st.session_state.selected_trail
else:
    selected_trail = sorted(set(df_segments["name"].tolist()))[0]
  

# Get the corresponding trail data for the selected trail
mask = (
    (df_trails["trail_name"] == selected_trail) &
    (df_trails["time_hour"] == pd.Timestamp(selected_time))
)

if mask.any():
    trail_row = df_trails.loc[mask].iloc[0]

    st.write(f"**Trail:** {selected_trail}")
    st.write(f"**Timestamp:** {trail_row['timestamp']}")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 🌧 Conditions")
        st.write(f"Overall: {trail_row['conditions']}")
        st.write(f"Water pooling: {trail_row['standing_water']}")
        st.write(f"Soft ground: {trail_row['soft_ground']}")
        st.write(f"Slippery roots: {trail_row['slippery_roots']}")
        st.write(f"Ice present: {trail_row['ice_present']}")

    with c2:
        st.markdown("### 🌡 Weather")
        features = trail_row["features"]
        st.write(f"Temperature: {features['temp_now']:.1f} °C")
        st.write(f"Rain 24h: {features['rain_24h']:.1f} mm")
        st.write(f"Humidity: {features['humidity_now']:.0f} %")
        st.write(f"Wind: {features['wind_now']:.1f} m/s")

    with c3:
        st.markdown("### 🏔 Terrain")
        st.write(f"Elevation: {features['elevation']:.0f} m")
        st.write(f"Slope: {features['slope']:.1f}°")

    with c4:
        st.markdown("### 🚵 Scores")
        st.write(f"Condition score: {trail_row['condition_score']:.1f}")
        st.write(f"Damage score: {trail_row['damage_score']:.1f}")
        st.write(f"Speed score: {trail_row['speed_score']:.1f}")

    if trail_row["condition_score"] <= 5:
        st.warning("⚠️ Trail likely slippery — not recommended to ride.")
    else:
        st.success("✅ Trail likely rideable.")
    if trail_row["damage_score"] >= 5:
        st.warning("⚠️ Trail likely vulnerable — riding may cause damage.")
    else:
        st.success("✅ Riding trail will likely not cause excessive damage.")

    st.divider()
else:
    selected_hour_str = pd.Timestamp(selected_time).strftime("%Y-%m-%d %H:%M")
    st.write(
        f"No data available for the selected trail at this hour "
        f"({selected_hour_str})"
    )
    

# --------------------------------------------------
# WARNING BANNER
# --------------------------------------------------


# --------------------------------------------------
# PLOTS
# --------------------------------------------------


def plot_trail_scores(df_areas, selected_area, selected_time=None):
    # ---- Filter area ----
    area_df = df_areas[df_areas["area_name"] == selected_area].copy()

    if area_df.empty:
        st.warning("No data available for selected area")
        return

    # ---- Ensure datetime ----
    area_df["timestamp"] = pd.to_datetime(area_df["timestamp"], errors="coerce")
    area_df = area_df.sort_values("timestamp")

    # ---- Extract series ----
    timestamps = area_df["timestamp"]
    condition_scores = area_df["avg_condition_score"]
    damage_scores = area_df["avg_damage_score"]

    # ---- Build figure ----
    fig = go.Figure()

    # ---- Fill between curves ----
    for i in range(1, len(area_df)):
        if damage_scores.iloc[i] > condition_scores.iloc[i]:
            fig.add_trace(go.Scatter( x=[timestamps.iloc[i-1], timestamps.iloc[i]], y=[damage_scores.iloc[i-1], damage_scores.iloc[i]], fill='tonexty', fillcolor='rgba(255,0,0,0.05)', 
                                     mode='lines', line=dict(width=0), showlegend=False )) 
            fig.add_trace(go.Scatter( x=[timestamps.iloc[i-1], timestamps.iloc[i]], y=[condition_scores.iloc[i-1], condition_scores.iloc[i]], fill='tonexty', fillcolor='rgba(255,0,0,0.05)', 
                                     mode='lines', line=dict(width=0), showlegend=False )) 
        elif damage_scores.iloc[i] < condition_scores.iloc[i]:
            fig.add_trace(go.Scatter( x=[timestamps.iloc[i-1], timestamps.iloc[i]], y=[condition_scores.iloc[i-1], condition_scores.iloc[i]], fill='tonexty', fillcolor='rgba(0,255,0,0.05)', 
                                    mode='lines', line=dict(width=0), showlegend=False )) 
            fig.add_trace(go.Scatter( x=[timestamps.iloc[i-1], timestamps.iloc[i]], y=[damage_scores.iloc[i-1], damage_scores.iloc[i]], fill='tonexty', fillcolor='rgba(0,255,0,0.05)', 
                                     mode='lines', line=dict(width=0), showlegend=False ))
    
    # Condition line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=condition_scores,
        mode="lines",
        name="Condition Score",
        line=dict(color="green", width=3)
    ))

    # Damage line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=damage_scores,
        mode="lines",
        name="Damage Score",
        line=dict(color="red", width=3)
    ))

    # ---- Optional speed score ----
    if "avg_speed_score" in area_df.columns:
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=area_df["avg_speed_score"],
            mode="lines",
            name="Speed Score",
            line=dict(color="#3498db", width=2, dash="dot")
        ))

    # ---- Selected time marker ----
    if selected_time is not None:
        fig.add_vline(
        x=pd.Timestamp(selected_time),
        line_width=2,
        line_dash="dash",
        line_color="red"
    )
    # Vertical time marker
    fig.add_vline(
        x=now,
        line_width=2,
        line_dash="dash",
        line_color="#333"
    )

    # ---- Layout ----
    fig.update_layout(
        title=f"{selected_area} – Area Scores Over Time",
        xaxis_title="Time",
        yaxis_title="Score",
        yaxis=dict(range=[0, 10], fixedrange=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=60, b=40)
    )

    fig.update_traces(line_shape="spline")

    st.plotly_chart(fig, use_container_width=True)

area_names = df_areas["area_name"].unique()
selected_area = st.selectbox("Select Area", area_names)

plot_trail_scores(
    df_areas,
    selected_area,
    selected_time=st.session_state.get("time_slider")
)


# --------------------------------------------------
# PLOT WEATHER DATA
# --------------------------------------------------
# Function to plot weather data for the selected area (Temperature, Wind, Rain)


def plot_weather_data(df_areas, selected_area, wind_threshold=10):

    # --- Filter & prepare data ---
    area_df = (
        df_areas[df_areas["area_name"] == selected_area]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if area_df.empty:
        st.warning(f"No weather data available for {selected_area}")
        return

    area_df["timestamp"] = pd.to_datetime(area_df["timestamp"])
    area_df["date"] = area_df["timestamp"].dt.date
    area_df["rain_accumulated"] = area_df.groupby("date")["rain_now"].cumsum()

    # --- Series ---
    timestamps = area_df["timestamp"]
    temps = area_df["temp_now"]
    winds = area_df["wind_now"]
    rains = area_df["rain_accumulated"]
    snow = area_df["snow_depth"] * 100  # cm

    fig = go.Figure()

    # -------- Temperature (segmented) --------
    temp_color = ['lightblue' if t <= 0 else '#FF6666' for t in temps]

    start = 0
    for i in range(1, len(temps)):
        if temp_color[i] != temp_color[start]:
            fig.add_trace(go.Scatter(
                x=timestamps.iloc[start:i+1],
                y=temps.iloc[start:i+1],
                mode="lines",
                line=dict(color=temp_color[start], width=3),
                name="Temperature (°C)" if start == 0 else None,
                showlegend=start == 0
            ))
            start = i

    fig.add_trace(go.Scatter(
        x=timestamps.iloc[start:],
        y=temps.iloc[start:],
        mode="lines",
        line=dict(color=temp_color[start], width=3),
        name="Temperature (°C)",
        showlegend=True
    ))

    # -------- Wind (segmented) --------
    wind_color = ['red' if w > wind_threshold else 'orange' for w in winds]

    start = 0
    for i in range(1, len(winds)):
        if wind_color[i] != wind_color[start]:
            fig.add_trace(go.Scatter(
                x=timestamps.iloc[start:i+1],
                y=winds.iloc[start:i+1],
                mode="lines",
                line=dict(color=wind_color[start], width=3),
                name="Wind (m/s)" if start == 0 else None,
                showlegend=start == 0
            ))
            start = i

    fig.add_trace(go.Scatter(
        x=timestamps.iloc[start:],
        y=winds.iloc[start:],
        mode="lines",
        line=dict(color=wind_color[start], width=3),
        name="Wind (m/s)",
        showlegend=True
    ))

    # -------- Snow --------
    fig.add_trace(go.Bar(
        x=timestamps,
        y=snow,
        name="Snow Depth (cm)",
        marker=dict(color="white"),
        opacity=0.4
    ))

    # -------- Rain --------
    fig.add_trace(go.Bar(
        x=timestamps,
        y=rains,
        name="Accumulated Rain (mm)",
        marker=dict(color="blue"),
        opacity=1.0
    ))

    # -------- Layout --------
    fig.update_layout(
        title=f"{selected_area} Weather Data Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        yaxis=dict(rangemode="tozero"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        barmode="overlay",
        showlegend=True
    )

    # Vertical time marker
    fig.add_vline(
        x=pd.Timestamp(selected_time),
        line_width=2,
        line_dash="dash",
        line_color="red"
    )
    # Vertical time marker
    fig.add_vline(
        x=now,
        line_width=2,
        line_dash="dash",
        line_color="#333"
    )

    st.plotly_chart(fig, use_container_width=True)



plot_weather_data(df_areas, selected_area)


# --------------------------------------------------
# PLOT CONDITION TYPES
# --------------------------------------------------
def plot_condition_types(df_areas, selected_area):

    area_df = (
        df_areas[df_areas["area_name"] == selected_area]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if area_df.empty:
        st.warning(f"No condition distribution data available for {selected_area}")
        return

    area_df["timestamp"] = pd.to_datetime(area_df["timestamp"])

    # Rename columns to readable labels
    df_dist = area_df.rename(columns={
        "pct_dry": "Dry",
        "pct_damp": "Damp",
        "pct_wet": "Wet",
        "pct_muddy": "Muddy",
        "pct_frost": "Frost",
    })[["timestamp", "Dry", "Damp", "Wet", "Muddy", "Frost"]]

    df_long = df_dist.melt(
        id_vars="timestamp",
        var_name="Condition",
        value_name="Percentage"
    )

    # --- Styling ---
    color_map = {
        "Dry":   "#2E7D32",  # dark green
        "Damp":  "#8BC34A",  # light green
        "Wet":   "#FFC107",  # yellow
        "Muddy": "#C62828",  # red
        "Frost": "#81D4FA",  # light blue
    }

    condition_order = ["Dry", "Damp", "Wet", "Muddy", "Frost"]

    fig = px.area(
        df_long,
        x="timestamp",
        y="Percentage",
        color="Condition",
        category_orders={"Condition": condition_order},
        color_discrete_map=color_map,
        labels={
            "timestamp": "Time",
            "Percentage": "Percentage of trails"
        }
    )

    fig.update_traces(
        opacity=0.25,
        line_shape="spline"
    )

    fig.update_layout(
        title=f"{selected_area} – Trail Condition Distribution",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        legend_title_text="Trail condition",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )

    # Selected time marker
    fig.add_vline(
        x=pd.Timestamp(selected_time),
        line_width=2,
        line_dash="dash",
        line_color="red"
    )
    # Vertical time marker
    fig.add_vline(
        x=now,
        line_width=2,
        line_dash="dash",
        line_color="#333"
    )

    st.plotly_chart(fig, use_container_width=True)
plot_condition_types(df_areas, selected_area)


# --------------------------------------------------
# PLOT TRAIL CONDITIONS
# --------------------------------------------------
def plot_trail_conditions(df_areas, selected_area):

    area_df = (
        df_areas[df_areas["area_name"] == selected_area]
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if area_df.empty:
        st.warning(f"No trail condition data available for {selected_area}")
        return

    area_df["timestamp"] = pd.to_datetime(area_df["timestamp"])

    timestamps = area_df["timestamp"]

    standing_water = area_df["pct_standing_water"]
    soft_ground = area_df["pct_soft_ground"]
    ice_present = area_df["pct_ice_present"]
    slippery_roots = area_df["pct_slippery_roots"]

    # Snow depth is usually meters → convert to cm for display
    snow_depth_cm = area_df["snow_depth"] * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=standing_water,
        mode="lines",
        name="Standing water (%)",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=soft_ground,
        mode="lines",
        name="Soft ground (%)",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=ice_present,
        mode="lines",
        name="Ice present (%)",
        line=dict(color="green")
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=slippery_roots,
        mode="lines",
        name="Slippery roots (%)",
        line=dict(color="red")
    ))

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=snow_depth_cm,
        mode="lines",
        name="Snow depth (cm)",
        line=dict(color="#AAAAAA", dash="dot")
    ))

    fig.update_layout(
        title=f"{selected_area} – Trail Conditions Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Percentage / cm",
        yaxis=dict(range=[0, 100], fixedrange=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        showlegend=True
    )

    fig.update_traces(line_shape="spline")

    fig.add_vline(
        x=pd.Timestamp(selected_time),
        line_width=2,
        line_dash="dash",
        line_color="red"
    )
    # Vertical time marker
    fig.add_vline(
        x=now,
        line_width=2,
        line_dash="dash",
        line_color="#333"
    )

    st.plotly_chart(fig, use_container_width=True)

plot_trail_conditions(df_areas, selected_area)


# --------------------------------------------------
# AREA SUMMARY
# --------------------------------------------------
st.subheader("Area comparison")

import pytz

# Define the timezone your data uses (adjust if different)
data_tz = "Europe/Oslo"

# Current time rounded to the hour, timezone-aware
now = pd.Timestamp.now(tz=data_tz).floor("h")
previous = now - pd.Timedelta(hours=1)

# Create columns for each area
area_cols = st.columns(len(df_areas['area_name'].unique()))

# Loop over areas
for col, area_name in zip(area_cols, df_areas['area_name'].unique()):
    # Filter rows for this area
    area_data = df_areas[df_areas['area_name'] == area_name].copy()

    # Ensure timestamp is timezone-aware and rounded to hour
    area_data['time_hour'] = pd.to_datetime(area_data['timestamp']).dt.tz_convert(data_tz).dt.floor("h")

    # Get current and previous hour stats
    stats_row = area_data[area_data['time_hour'] == now]
    previous_row = area_data[area_data['time_hour'] == previous]

    # Skip if no data
    if stats_row.empty:
        continue
    if previous_row.empty:
        # fallback: use the earliest available data for delta calculations
        previous_row = stats_row

    # Extract series as dict
    stats = stats_row.iloc[0].to_dict()
    previous_stats = previous_row.iloc[0].to_dict()

    with col:
        st.markdown(f"### {area_name}")
        condition_order = ["Dry", "Damp", "Wet", "Muddy", "Frost"]

        condition_df = pd.DataFrame({
            "Condition": condition_order,
            "Percent": [
                stats["pct_dry"],
                stats["pct_damp"],
                stats["pct_wet"],
                stats["pct_muddy"],
                stats["pct_frost"],
            ],
            "Delta": [
                stats["pct_dry"] - previous_stats["pct_dry"],
                stats["pct_damp"] - previous_stats["pct_damp"],
                stats["pct_wet"] - previous_stats["pct_wet"],
                stats["pct_muddy"] - previous_stats["pct_muddy"],
                stats["pct_frost"] - previous_stats["pct_frost"],
            ],
        })

        condition_df["Condition"] = pd.Categorical(
            condition_df["Condition"],
            categories=condition_order,
            ordered=True
        )
        condition_df = condition_df.sort_values("Condition")

        # Pie chart
        color_map = {
            "Dry":   "#2E7D32",  # dark green
            "Damp":  "#8BC34A",  # light green
            "Wet":   "#FFC107",  # yellow / warning
            "Muddy": "#C62828",  # red
            "Frost": "#81D4FA",  # light blue
        }

        fig = go.Figure(
            go.Pie(
                labels=condition_df["Condition"],
                values=condition_df["Percent"],
                hole=0.4,
                sort=False
            )
        )

        fig.update_traces(
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:.0f}%<br>Δ %{customdata:.0f}%",
            customdata=condition_df["Delta"],
            marker=dict(colors=[color_map[c] for c in condition_df["Condition"]])
        )

        fig.update_layout(
            height=200,
            margin=dict(t=30, b=30, l=5, r=5),
            showlegend=True
        )

        st.plotly_chart(fig, config={"width":"stretch"}, key=area_name)

        # Display key metrics
        st.metric("Water pooling", f"{stats['pct_standing_water']:.0f}%", 
                  delta=f"{stats['pct_standing_water']-previous_stats['pct_standing_water']:.0f}%")
        st.metric("Soft ground", f"{stats['pct_soft_ground']:.0f}%", 
                  delta=f"{stats['pct_soft_ground']-previous_stats['pct_soft_ground']:.0f}%")
        st.metric("Slippery roots", f"{stats['pct_slippery_roots']:.0f}%", 
                  delta=f"{stats['pct_slippery_roots']-previous_stats['pct_slippery_roots']:.0f}%")
        st.metric("Snow depth", f"{stats['snow_depth']:.0f}cm", 
                  delta=f"{stats['snow_depth']-previous_stats['snow_depth']:.0f}cm")
        st.metric("Icy", f"{stats['pct_ice_present']:.0f}%", 
                  delta=f"{stats['pct_ice_present']-previous_stats['pct_ice_present']:.0f}%")
        st.metric("Damage score", f"{stats['avg_damage_score']:.1f}", 
                  delta=f"{stats['avg_damage_score']-previous_stats['avg_damage_score']:.1f}")
        st.metric("Trail condition score", f"{stats['avg_condition_score']:.1f}", 
                  delta=f"{stats['avg_condition_score']-previous_stats['avg_condition_score']:.1f}")
        st.metric("Predicted speed vs normal", f"{stats.get('relative_speed_percentage', 100):.1f}%", 
                  delta=f"{stats.get('relative_speed_percentage', 100)-previous_stats.get('relative_speed_percentage', 100):.1f}%")

st.divider()


# --------------------------------------------------
# AREA OVERVIEW TABLE
# --------------------------------------------------
st.subheader("Area overview")

st.dataframe(
    df_areas,
    width='stretch',
    hide_index=True,
)

st.divider()

# --------------------------------------------------
# TRAIL OVERVIEW TABLE
# --------------------------------------------------
st.subheader("Trail overview")

st.dataframe(
    df_trails,
    width='stretch',
    hide_index=True,
)

st.divider()



