import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

def detect_hotspots(df, eps=0.5, min_samples=10):
    """Detect accident hotspots using DBSCAN clustering"""
    coords = df[['Start_Lat', 'Start_Lng']].values
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    df_copy = df.copy()
    df_copy['Cluster'] = clustering.fit_predict(np.radians(coords))
    
    hotspots = df_copy[df_copy['Cluster'] != -1].groupby('Cluster').agg({
        'Start_Lat': 'mean',
        'Start_Lng': 'mean',
        'ID': 'count',
        'Severity': 'mean'
    }).reset_index()
    
    hotspots.columns = ['Cluster', 'Latitude', 'Longitude', 'Accident_Count', 'Avg_Severity']
    hotspots = hotspots.sort_values('Accident_Count', ascending=False)
    
    return hotspots, df_copy

def calculate_risk_score(row):
    """Calculate risk score based on multiple factors"""
    severity_weight = row.get('Severity', 2) * 0.4
    weather_score = 0.2 if row.get('Weather_Condition', 'Clear') in ['Rain', 'Snow', 'Fog', 'Heavy Rain'] else 0
    visibility_score = 0.2 if row.get('Visibility(mi)', 10) < 2 else 0
    hour_score = 0.2 if row.get('Hour', 12) in [0, 1, 2, 3, 4, 5, 22, 23] else 0
    
    return severity_weight + weather_score + visibility_score + hour_score