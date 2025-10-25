import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium

def find_accidents_on_route(df, start_coords, end_coords, radius_miles=5):
    """Find accidents along a route within specified radius"""
    accidents_on_route = []
    
    for idx, row in df.iterrows():
        accident_coords = (row['Start_Lat'], row['Start_Lng'])
        
        dist_to_start = geodesic(start_coords, accident_coords).miles
        dist_to_end = geodesic(end_coords, accident_coords).miles
        total_route_distance = geodesic(start_coords, end_coords).miles
        
        if dist_to_start + dist_to_end <= total_route_distance + radius_miles:
            accidents_on_route.append(row)
    
    return pd.DataFrame(accidents_on_route) if accidents_on_route else pd.DataFrame()

def calculate_route_safety_score(route_accidents):
    """Calculate safety score for a route"""
    if route_accidents.empty:
        return 95.0
    
    accident_count = len(route_accidents)
    avg_severity = route_accidents['Severity'].mean()
    
    base_score = 100
    accident_penalty = min(accident_count * 2, 50)
    severity_penalty = (avg_severity - 1) * 10
    
    safety_score = max(base_score - accident_penalty - severity_penalty, 0)
    return round(safety_score, 1)

def create_route_map(df, start_coords, end_coords, route_accidents):
    """Create map showing route and accidents"""
    center_lat = (start_coords[0] + end_coords[0]) / 2
    center_lng = (start_coords[1] + end_coords[1]) / 2
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=8)
    
    folium.Marker(
        start_coords,
        popup="<b>Start</b>",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        end_coords,
        popup="<b>Destination</b>",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    folium.PolyLine(
        [start_coords, end_coords],
        color='blue',
        weight=3,
        opacity=0.7
    ).add_to(m)
    
    if not route_accidents.empty:
        for idx, row in route_accidents.iterrows():
            folium.CircleMarker(
                location=[row['Start_Lat'], row['Start_Lng']],
                radius=5,
                popup=f"<b>Severity {row['Severity']}</b><br>{row['City']}",
                color='orange' if row['Severity'] < 3 else 'red',
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
    
    return m