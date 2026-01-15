import pandas as pd

def calculate_condition_score(df):
    c_conditions_map = {"Dry": 2, "Damp": 1, "Wet": 0.5, "Muddy": 0, "Frost": 2}
    c_standing_water_map = {False: 1, True: 0}
    c_slippery_roots_map = {False: 2, True: 0}
    c_ice_present_map = {False: 3, True: 0}

    df["condition_score"] = (
        df["conditions"].map(c_conditions_map).fillna(0) +
        df["standing_water"].map(c_standing_water_map).fillna(0) +
        df["slippery_roots"].map(c_slippery_roots_map).fillna(0) +
        df["ice_present"].map(c_ice_present_map).fillna(0)
    )
    df["condition_score"] = df["condition_score"].round(1)

    return df


def calculate_damage_score(df):
    d_conditions_map = {"Dry": 1, "Damp": 2, "Wet": 3, "Muddy": 5, "Frost": 0}
    d_standing_water_map = {False: 0, True: 3}
    d_soft_ground_map = {False: 0, True: 2}

    df["damage_score"] = (
        df["conditions"].map(d_conditions_map).fillna(0) + 
        df["standing_water"].map(d_standing_water_map).fillna(0) +
        df["soft_ground"].map(d_soft_ground_map).fillna(0)
    )

    df["damage_score"] = df["damage_score"].round(1)

    return df


def calculate_median_speed_score(df):
    # Calculate the median speed for each trail_name
    trail_medians = df.groupby("trail_name")["median_speed"].median().to_dict()

    # Function to calculate a score from 0 to 10 based on deviation from the trail's median
    def score_from_speed(row):
        trail_name = row["trail_name"]
        median_speed = row["median_speed"]
        trail_median = trail_medians[trail_name]

        # Calculate the score based on the deviation from the median speed
        # Here we use simple scaling: higher than median -> closer to 10, lower -> closer to 0
        if median_speed == 0:
            return 0  # Avoid division by zero (this case should not occur, but for safety)
        
        # Normalize speed as a score (here we assume 1.0x is the 'normal' speed)
        score = ((median_speed / trail_median) - 1) * 5 + 5

        # Bound the score between 0 and 10
        return min(max(score, 0), 10)

    # Apply the score function to each row in the DataFrame
    df["speed_score"] = df.apply(score_from_speed, axis=1)

    return df

def calculate_area_scores(df_areas, df_trails):
    df_area_scores = df_trails.groupby(['area_id', 'timestamp']).agg(
        avg_damage_score=('damage_score', 'mean'),
        avg_condition_score=('condition_score', 'mean'),
        avg_speed_score=('speed_score', 'mean')
    ).reset_index()

    df_areas = df_areas.merge(df_area_scores, on=['area_id', 'timestamp'], how='left')
    return df_areas
