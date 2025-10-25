import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd

def create_basic_map(df, limit=1000):
    """Create basic map with accident markers"""
    center_lat = df['Start_Lat'].mean()
    center_lng = df['Start_Lng'].mean()
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=4, tiles='OpenStreetMap')
    
    marker_cluster = MarkerCluster().add_to(m)
    
    sample_df = df.head(limit)
    for idx, row in sample_df.iterrows():
        folium.CircleMarker(
            location=[row['Start_Lat'], row['Start_Lng']],
            radius=3,
            popup=f"<b>Severity:</b> {row['Severity']}<br><b>City:</b> {row['City']}<br><b>Weather:</b> {row['Weather_Condition']}",
            color='red' if row['Severity'] >= 3 else 'orange' if row['Severity'] == 2 else 'yellow',
            fill=True,
            fillOpacity=0.6
        ).add_to(marker_cluster)
    
    return m

def create_heatmap(df, limit=5000):
    """Create heatmap of accident density"""
    center_lat = df['Start_Lat'].mean()
    center_lng = df['Start_Lng'].mean()
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=4, tiles='OpenStreetMap')
    
    heat_data = [[row['Start_Lat'], row['Start_Lng']] for idx, row in df.head(limit).iterrows()]
    HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)
    
    return m

def create_severity_map(df, limit=1000):
    """Create map colored by severity"""
    center_lat = df['Start_Lat'].mean()
    center_lng = df['Start_Lng'].mean()
    
    m = folium.Map(location=[center_lat, center_lng], zoom_start=4, tiles='CartoDB positron')
    
    color_map = {1: 'green', 2: 'yellow', 3: 'orange', 4: 'red'}
    
    for severity in sorted(df['Severity'].unique()):
        severity_df = df[df['Severity'] == severity].head(limit // 4)
        for idx, row in severity_df.iterrows():
            folium.CircleMarker(
                location=[row['Start_Lat'], row['Start_Lng']],
                radius=4,
                popup=f"<b>Severity {severity}</b><br>{row['City']}, {row['State']}",
                color=color_map.get(severity, 'gray'),
                fill=True,
                fillOpacity=0.7
            ).add_to(m)
    
    return m
