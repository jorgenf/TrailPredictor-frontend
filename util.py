# util.py
import pandas as pd

def calculate_condition_score(df: pd.DataFrame) -> pd.DataFrame:
    c_conditions_map = {"Dry": 2, "Damp": 1, "Wet": 0.5, "Muddy": 0, "Frost": 2}
    c_standing_water_map = {False: 1, True: 0}
    c_slippery_roots_map = {False: 2, True: 0}
    c_ice_present_map = {False: 3, True: 0}

    df["condition_score"] = (
        df["conditions"].map(c_conditions_map).fillna(0) +
        df["standing_water"].fillna(False).map(c_standing_water_map) +
        df["slippery_roots"].fillna(False).map(c_slippery_roots_map) +
        df["ice_present"].fillna(False).map(c_ice_present_map)
    ).fillna(0).round(1)

    return df


def calculate_damage_score(df: pd.DataFrame) -> pd.DataFrame:
    d_conditions_map = {"Dry": 1, "Damp": 2, "Wet": 3, "Muddy": 5, "Frost": 0}
    d_standing_water_map = {False: 0, True: 3}
    d_soft_ground_map = {False: 0, True: 2}

    df["damage_score"] = (
        df["conditions"].map(d_conditions_map).fillna(0) +
        df["standing_water"].fillna(False).map(d_standing_water_map) +
        df["soft_ground"].fillna(False).map(d_soft_ground_map)
    ).fillna(0).round(1)

    return df


def calculate_median_speed_score(df: pd.DataFrame) -> pd.DataFrame:
    if "trail_name" not in df.columns or "median_speed" not in df.columns:
        df["speed_score"] = 0
        return df

    # compute median speeds per trail, ignoring missing names
    trail_medians = df.dropna(subset=["trail_name"]).groupby("trail_name")["median_speed"].median().to_dict()

    def score_from_speed(row):
        trail_name = row.get("trail_name")
        median_speed = row.get("median_speed", 0)

        # Skip missing trail_name or zero speed
        if not trail_name or median_speed == 0:
            return 0

        trail_median = trail_medians.get(trail_name, median_speed)  # fallback to current row speed
        if trail_median == 0:
            return 0

        score = ((median_speed / trail_median) - 1) * 5 + 5
        return min(max(score, 0), 10)

    df["speed_score"] = df.apply(score_from_speed, axis=1)
    return df


def calculate_area_scores(df_areas: pd.DataFrame, df_trails: pd.DataFrame) -> pd.DataFrame:
    if df_areas.empty or df_trails.empty:
        return df_areas

    df_area_scores = df_trails.groupby(['area_id', 'timestamp']).agg(
        avg_damage_score=('damage_score', 'mean'),
        avg_condition_score=('condition_score', 'mean'),
        avg_speed_score=('speed_score', 'mean')
    ).reset_index()

    df_areas = df_areas.merge(df_area_scores, on=['area_id', 'timestamp'], how='left')
    return df_areas
