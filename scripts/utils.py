import pandas as pd

def get_city_coordinates(city_name, df):
    """Get average coordinates for a city"""
    city_data = df[df['City'] == city_name]
    if not city_data.empty:
        return (city_data['Start_Lat'].mean(), city_data['Start_Lng'].mean())
    return None

def format_number(num):
    """Format large numbers with commas"""
    return f"{num:,}"

def get_summary_statistics(df):
    """Get summary statistics for dashboard"""
    stats = {
        'total_accidents': len(df),
        'avg_severity': round(df['Severity'].mean(), 2),
        'most_common_weather': df['Weather_Condition'].mode()[0] if not df.empty else 'N/A',
        'most_dangerous_city': df['City'].value_counts().index[0] if not df.empty else 'N/A',
        'peak_hour': df['Hour'].mode()[0] if not df.empty else 'N/A',
        'high_severity_count': len(df[df['Severity'] >= 3])
    }
    return stats