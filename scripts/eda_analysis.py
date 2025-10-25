import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

plt.style.use("seaborn-v0_8-darkgrid")

def plot_severity(df):
    """Plot severity distribution"""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(x='Severity', data=df, palette='coolwarm', ax=ax)
    ax.set_title("Accident Severity Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Severity Level")
    ax.set_ylabel("Count")
    return fig

def plot_daywise(df):
    """Plot accidents by day of week"""
    fig, ax = plt.subplots(figsize=(10, 4))
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.countplot(x='DayOfWeek', data=df, order=order, palette='mako', ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Accidents per Day of Week", fontsize=14, fontweight='bold')
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Number of Accidents")
    return fig

def plot_hourly(df):
    """Plot accidents by hour of day"""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['Hour'], bins=24, color='orange', kde=True, ax=ax)
    ax.set_title("Accidents by Hour of Day", fontsize=14, fontweight='bold')
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("Number of Accidents")
    return fig

def plot_top_cities(df, top_n=10):
    """Plot top cities with most accidents"""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_cities = df['City'].value_counts().head(top_n)
    sns.barplot(x=top_cities.values, y=top_cities.index, palette='viridis', ax=ax)
    ax.set_title(f"Top {top_n} Cities with Most Accidents", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Accidents")
    ax.set_ylabel("City")
    return fig

def plot_weather_vs_severity(df):
    """Plot severity by weather condition"""
    fig, ax = plt.subplots(figsize=(12, 5))
    top_weather = df['Weather_Condition'].value_counts().head(8).index
    sns.boxplot(data=df[df['Weather_Condition'].isin(top_weather)],
                x='Weather_Condition', y='Severity', palette='rocket', ax=ax)
    plt.xticks(rotation=45, ha='right')
    ax.set_title("Severity by Weather Condition", fontsize=14, fontweight='bold')
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Severity Level")
    return fig

def plot_monthly_trend(df):
    """Plot monthly accident trends"""
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly = df.groupby(['Year', 'Month']).size().reset_index(name='Count')
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly['Month'] = pd.Categorical(monthly['Month'], categories=month_order, ordered=True)
    monthly = monthly.sort_values(['Year', 'Month'])
    
    for year in monthly['Year'].unique():
        year_data = monthly[monthly['Year'] == year]
        ax.plot(year_data['Month'], year_data['Count'], marker='o', label=f'Year {year}')
    
    plt.xticks(rotation=45, ha='right')
    ax.set_title("Monthly Accident Trends", fontsize=14, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Accidents")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_state_distribution(df, top_n=15):
    """Plot accidents by state"""
    fig, ax = plt.subplots(figsize=(12, 6))
    top_states = df['State'].value_counts().head(top_n)
    sns.barplot(x=top_states.values, y=top_states.index, palette='plasma', ax=ax)
    ax.set_title(f"Top {top_n} States with Most Accidents", fontsize=14, fontweight='bold')
    ax.set_xlabel("Number of Accidents")
    ax.set_ylabel("State")
    return fig

def plot_correlation_heatmap(df):
    """Plot correlation heatmap for numerical features"""
    fig, ax = plt.subplots(figsize=(10, 8))
    numerical_cols = ['Severity', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
                      'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Hour']
    corr_data = df[numerical_cols].dropna().corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    return fig
