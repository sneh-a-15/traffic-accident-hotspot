import pandas as pd
import numpy as np
from geopy.distance import geodesic
import folium
from scipy.spatial import cKDTree

def find_accidents_on_route(df, start_coords, end_coords, radius_miles=5):
    """
    Find accidents along a route within specified radius
    Optimized using spatial indexing for large datasets
    """
    # Convert miles to degrees (approximate)
    radius_degrees = radius_miles / 69.0  # 1 degree ≈ 69 miles
    
    # Create bounding box for initial filtering
    min_lat = min(start_coords[0], end_coords[0]) - radius_degrees
    max_lat = max(start_coords[0], end_coords[0]) + radius_degrees
    min_lng = min(start_coords[1], end_coords[1]) - radius_degrees
    max_lng = max(start_coords[1], end_coords[1]) + radius_degrees
    
    # Filter by bounding box first (much faster)
    bbox_filter = (
        (df['Start_Lat'] >= min_lat) & (df['Start_Lat'] <= max_lat) &
        (df['Start_Lng'] >= min_lng) & (df['Start_Lng'] <= max_lng)
    )
    candidates = df[bbox_filter].copy()
    
    if len(candidates) == 0:
        return pd.DataFrame()
    
    # Use vectorized operations for distance calculations
    total_route_distance = geodesic(start_coords, end_coords).miles
    
    # Calculate distances for all candidates at once
    accidents_on_route = []
    
    # For large datasets, use spatial indexing
    if len(candidates) > 10000:
        # Use KD-Tree for efficient nearest neighbor search
        coords = np.radians(candidates[['Start_Lat', 'Start_Lng']].values)
        tree = cKDTree(coords)
        
        # Sample points along the route
        num_points = int(total_route_distance / 5) + 1  # Every 5 miles
        route_points = []
        for i in range(num_points + 1):
            t = i / num_points
            lat = start_coords[0] + t * (end_coords[0] - start_coords[0])
            lng = start_coords[1] + t * (end_coords[1] - start_coords[1])
            route_points.append([lat, lng])
        
        route_points = np.radians(route_points)
        
        # Find accidents near route points
        radius_rad = radius_miles / 3959.0  # Earth radius in miles
        nearby_indices = set()
        
        for point in route_points:
            indices = tree.query_ball_point(point, radius_rad)
            nearby_indices.update(indices)
        
        accidents_on_route = candidates.iloc[list(nearby_indices)]
    else:
        # For smaller datasets, use the original method
        for idx, row in candidates.iterrows():
            accident_coords = (row['Start_Lat'], row['Start_Lng'])
            
            dist_to_start = geodesic(start_coords, accident_coords).miles
            dist_to_end = geodesic(end_coords, accident_coords).miles
            
            # Check if point is near the route
            if dist_to_start + dist_to_end <= total_route_distance + radius_miles:
                accidents_on_route.append(row)
        
        accidents_on_route = pd.DataFrame(accidents_on_route) if accidents_on_route else pd.DataFrame()
    
    return accidents_on_route


def calculate_route_safety_score(route_accidents):
    """
    Calculate safety score for a route (0-100, higher is safer)
    FIXED: Now only takes route_accidents as parameter
    """
    if route_accidents.empty:
        return 95.0
    
    accident_count = len(route_accidents)
    avg_severity = route_accidents['Severity'].mean()
    
    # More sophisticated scoring
    base_score = 100
    
    # Accident count penalty (logarithmic scale)
    if accident_count > 0:
        accident_penalty = min(10 * np.log10(accident_count + 1), 40)
    else:
        accident_penalty = 0
    
    # Severity penalty
    severity_penalty = (avg_severity - 1) * 10
    
    # High severity accidents get extra penalty
    high_severity_count = len(route_accidents[route_accidents['Severity'] >= 3])
    high_severity_penalty = min(high_severity_count * 5, 20)
    
    safety_score = max(base_score - accident_penalty - severity_penalty - high_severity_penalty, 0)
    return round(safety_score, 1)


def get_route_statistics(route_accidents):
    """Get detailed statistics about accidents on route"""
    if route_accidents.empty:
        return {
            'total_accidents': 0,
            'avg_severity': 0,
            'severity_distribution': {},
            'weather_distribution': {},
            'time_distribution': {}
        }
    
    stats = {
        'total_accidents': len(route_accidents),
        'avg_severity': round(route_accidents['Severity'].mean(), 2),
        'severity_distribution': route_accidents['Severity'].value_counts().to_dict(),
        'weather_distribution': route_accidents['Weather_Condition'].value_counts().head(5).to_dict() if 'Weather_Condition' in route_accidents.columns else {},
        'time_distribution': route_accidents['Hour'].value_counts().sort_index().to_dict() if 'Hour' in route_accidents.columns else {},
        'most_dangerous_hour': int(route_accidents['Hour'].mode()[0]) if 'Hour' in route_accidents.columns and len(route_accidents) > 0 else None
    }
    
    return stats


def create_route_map(df, start_coords, end_coords, route_accidents):
    """Create interactive map showing route and accidents"""
    center_lat = (start_coords[0] + end_coords[0]) / 2
    center_lng = (start_coords[1] + end_coords[1]) / 2
    
    # Calculate zoom level based on distance
    distance = geodesic(start_coords, end_coords).miles
    if distance < 50:
        zoom = 10
    elif distance < 100:
        zoom = 9
    elif distance < 200:
        zoom = 8
    else:
        zoom = 7
    
    m = folium.Map(
        location=[center_lat, center_lng], 
        zoom_start=zoom,
        tiles='CartoDB positron'
    )
    
    # Add start marker
    folium.Marker(
        start_coords,
        popup="<b>Start Location</b>",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    # Add end marker
    folium.Marker(
        end_coords,
        popup="<b>Destination</b>",
        icon=folium.Icon(color='blue', icon='stop', prefix='fa')
    ).add_to(m)
    
    # Draw route line
    folium.PolyLine(
        [start_coords, end_coords],
        color='#2E86AB',
        weight=4,
        opacity=0.8,
        popup=f"<b>Route</b><br>Distance: {geodesic(start_coords, end_coords).miles:.1f} miles"
    ).add_to(m)
    
    # Add accidents with color coding by severity
    if not route_accidents.empty:
        severity_colors = {
            1: '#90EE90',  # Light green
            2: '#FFD700',  # Gold
            3: '#FFA500',  # Orange
            4: '#FF4500'   # Red-orange
        }
        
        # Limit to most severe accidents if too many
        display_accidents = route_accidents.nlargest(500, 'Severity') if len(route_accidents) > 500 else route_accidents
        
        for idx, row in display_accidents.iterrows():
            severity = row['Severity']
            color = severity_colors.get(severity, '#808080')
            
            popup_text = f"""
            <b>Accident Details</b><br>
            <b>Severity:</b> {severity}<br>
            <b>Location:</b> {row['City']}<br>
            <b>Weather:</b> {row.get('Weather_Condition', 'Unknown')}<br>
            <b>Time:</b> {row.get('Start_Time', 'Unknown')}
            """
            
            folium.CircleMarker(
                location=[row['Start_Lat'], row['Start_Lng']],
                radius=3 + severity,
                popup=folium.Popup(popup_text, max_width=250),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=1
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 160px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin:0; font-weight:bold;">Accident Severity</p>
    <p style="margin:5px 0;"><span style="color:#90EE90;">●</span> Level 1 (Minor)</p>
    <p style="margin:5px 0;"><span style="color:#FFD700;">●</span> Level 2 (Moderate)</p>
    <p style="margin:5px 0;"><span style="color:#FFA500;">●</span> Level 3 (Serious)</p>
    <p style="margin:5px 0;"><span style="color:#FF4500;">●</span> Level 4 (Severe)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


def compare_routes(df, routes):
    """
    Compare multiple routes and rank them by safety
    
    Parameters:
    -----------
    df : pd.DataFrame
        Accident data
    routes : list of tuples
        List of (start_coords, end_coords, route_name) tuples
    
    Returns:
    --------
    pd.DataFrame with route comparisons
    """
    results = []
    
    for start_coords, end_coords, route_name in routes:
        route_accidents = find_accidents_on_route(df, start_coords, end_coords)
        safety_score = calculate_route_safety_score(route_accidents)
        distance = geodesic(start_coords, end_coords).miles
        
        results.append({
            'Route': route_name,
            'Distance_Miles': round(distance, 1),
            'Accident_Count': len(route_accidents),
            'Safety_Score': safety_score,
            'Avg_Severity': round(route_accidents['Severity'].mean(), 2) if not route_accidents.empty else 0,
            'Recommendation': 'Recommended' if safety_score >= 70 else 'Use Caution' if safety_score >= 50 else 'High Risk'
        })
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('Safety_Score', ascending=False)
    
    return comparison_df


def get_dangerous_segments(df, start_coords, end_coords, segment_length_miles=10):
    """
    Divide route into segments and identify most dangerous ones
    
    Parameters:
    -----------
    df : pd.DataFrame
        Accident data
    start_coords : tuple
        (lat, lng) of start
    end_coords : tuple
        (lat, lng) of end
    segment_length_miles : int
        Length of each segment in miles
    
    Returns:
    --------
    List of segments with accident counts
    """
    total_distance = geodesic(start_coords, end_coords).miles
    num_segments = max(int(total_distance / segment_length_miles), 1)
    
    segments = []
    for i in range(num_segments):
        t1 = i / num_segments
        t2 = (i + 1) / num_segments
        
        seg_start = (
            start_coords[0] + t1 * (end_coords[0] - start_coords[0]),
            start_coords[1] + t1 * (end_coords[1] - start_coords[1])
        )
        seg_end = (
            start_coords[0] + t2 * (end_coords[0] - start_coords[0]),
            start_coords[1] + t2 * (end_coords[1] - start_coords[1])
        )
        
        seg_accidents = find_accidents_on_route(df, seg_start, seg_end, radius_miles=3)
        
        segments.append({
            'Segment': i + 1,
            'Start': seg_start,
            'End': seg_end,
            'Accident_Count': len(seg_accidents),
            'Avg_Severity': round(seg_accidents['Severity'].mean(), 2) if not seg_accidents.empty else 0
        })
    
    return pd.DataFrame(segments).sort_values('Accident_Count', ascending=False)