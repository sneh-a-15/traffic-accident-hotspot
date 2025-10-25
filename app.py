import streamlit as st
import pandas as pd
from scripts import data_loader, eda_analysis, map_generator, hotspot_detector, route_analyzer, prediction_model, utils
from streamlit_folium import st_folium
import folium
import matplotlib.pyplot as plt

DATA_PATH = "data/US_Accidents_March23.csv"

st.set_page_config(page_title="Traffic Accident Hotspot", page_icon="🚦", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚦 Traffic Accident Hotspot Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Comprehensive Analysis & Prediction System</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")
data_limit = st.sidebar.slider("Data Sample Size", 1000, 50000, 10000, 1000)

# Load data
with st.spinner("Loading data..."):
    df = data_loader.load_data(DATA_PATH, limit=data_limit)

# Sidebar filters
st.sidebar.header("🔍 Filters")
cities = ['All'] + sorted(df['City'].unique().tolist())
selected_city = st.sidebar.selectbox("Select City", cities)

severity_levels = ['All'] + sorted(df['Severity'].unique().tolist())
selected_severity = st.sidebar.selectbox("Select Severity", severity_levels)

# Apply filters
filtered_df = df.copy()
if selected_city != 'All':
    filtered_df = filtered_df[filtered_df['City'] == selected_city]
if selected_severity != 'All':
    filtered_df = filtered_df[filtered_df['Severity'] == selected_severity]

st.sidebar.metric("Filtered Records", utils.format_number(len(filtered_df)))

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard", 
    "🗺️ Interactive Maps", 
    "🛣️ Route Analysis", 
    "🤖 ML Prediction", 
    "🎯 Hotspot Detection"
])

# TAB 1: Dashboard
with tab1:
    st.header("📊 Exploratory Data Analysis")
    
    # Summary metrics
    stats = utils.get_summary_statistics(filtered_df)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Accidents", utils.format_number(stats['total_accidents']))
    with col2:
        st.metric("Average Severity", stats['avg_severity'])
    with col3:
        st.metric("High Severity (3+)", utils.format_number(stats['high_severity_count']))
    with col4:
        st.metric("Peak Hour", f"{stats['peak_hour']}:00")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Severity Distribution")
        st.pyplot(eda_analysis.plot_severity(filtered_df))
        
        st.subheader("Accidents by Day of Week")
        st.pyplot(eda_analysis.plot_daywise(filtered_df))
    
    with col2:
        st.subheader("Hourly Distribution")
        st.pyplot(eda_analysis.plot_hourly(filtered_df))
        
        st.subheader("Top 10 Cities")
        st.pyplot(eda_analysis.plot_top_cities(filtered_df))
    
    st.subheader("Weather vs Severity")
    st.pyplot(eda_analysis.plot_weather_vs_severity(filtered_df))
    
    st.subheader("State Distribution")
    st.pyplot(eda_analysis.plot_state_distribution(filtered_df))
    
    st.subheader("Feature Correlation")
    st.pyplot(eda_analysis.plot_correlation_heatmap(filtered_df))

# TAB 2: Interactive Maps
with tab2:
    st.header("🗺️ Interactive Accident Maps")
    
    map_type = st.radio("Select Map Type", ["Basic Markers", "Heatmap", "Severity Map"], horizontal=True)
    map_limit = st.slider("Number of Points to Display", 100, 5000, 1000, 100)
    
    # Generate map only when button is clicked
    if st.button("🗺️ Generate Map", type="primary"):
        with st.spinner("Generating map..."):
            if map_type == "Basic Markers":
                m = map_generator.create_basic_map(filtered_df, limit=map_limit)
            elif map_type == "Heatmap":
                m = map_generator.create_heatmap(filtered_df, limit=map_limit)
            else:
                m = map_generator.create_severity_map(filtered_df, limit=map_limit)
            
            st.session_state['current_map'] = m
            st.session_state['map_type'] = map_type
    
    # Display map if it exists in session state
    if 'current_map' in st.session_state:
        st.info(f"Displaying: {st.session_state.get('map_type', 'Map')}")
        st_folium(st.session_state['current_map'], width=1400, height=600, returned_objects=[])

# TAB 3: Route Analysis
with tab3:
    st.header("🛣️ Route Safety Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Start Location")
        start_city = st.selectbox("Select Start City", sorted(df['City'].unique()))
        start_coords = utils.get_city_coordinates(start_city, df)
        if start_coords:
            st.info(f"📍 Coordinates: {start_coords[0]:.4f}, {start_coords[1]:.4f}")
    
    with col2:
        st.subheader("Destination")
        end_city = st.selectbox("Select Destination City", sorted(df['City'].unique()))
        end_coords = utils.get_city_coordinates(end_city, df)
        if end_coords:
            st.info(f"📍 Coordinates: {end_coords[0]:.4f}, {end_coords[1]:.4f}")
    
    search_radius = st.slider("Search Radius (miles)", 5, 50, 10)
    
    if st.button("🔍 Analyze Route", type="primary"):
        if start_coords and end_coords:
            with st.spinner("Analyzing route..."):
                route_accidents = route_analyzer.find_accidents_on_route(
                    df, start_coords, end_coords, radius_miles=search_radius
                )
                
                safety_score = route_analyzer.calculate_route_safety_score(route_accidents)
                route_map = route_analyzer.create_route_map(df, start_coords, end_coords, route_accidents)
                
                # Store results in session state
                st.session_state['route_accidents'] = route_accidents
                st.session_state['route_safety_score'] = safety_score
                st.session_state['route_map'] = route_map
                st.session_state['route_analyzed'] = True
        else:
            st.error("Could not find coordinates for selected cities")
    
    # Display results if they exist in session state
    if st.session_state.get('route_analyzed', False):
        route_accidents = st.session_state['route_accidents']
        safety_score = st.session_state['route_safety_score']
        route_map = st.session_state['route_map']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accidents on Route", len(route_accidents))
        with col2:
            st.metric("Safety Score", f"{safety_score}/100")
        with col3:
            if not route_accidents.empty:
                avg_sev = route_accidents['Severity'].mean()
                st.metric("Avg Severity", f"{avg_sev:.2f}")
            else:
                st.metric("Avg Severity", "N/A")
        
        st_folium(route_map, width=1400, height=600, key="route_map")
        
        if not route_accidents.empty:
            st.subheader("Accidents on Route")
            st.dataframe(route_accidents[['City', 'Severity', 'Weather_Condition', 'Start_Time']].head(20))

# TAB 4: ML Prediction
with tab4:
    st.header("🤖 Accident Severity Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Train Model")
        model_type = st.selectbox("Select Model", ["RandomForest", "GradientBoosting"])
        
        if st.button("🎯 Train Model", type="primary"):
            with st.spinner("Training model..."):
                ml_df = prediction_model.preprocess_for_ml(df)
                model, acc, X_test, y_test, preds, feature_names = prediction_model.train_model(ml_df, model_type)
                
                st.session_state['model'] = model
                st.session_state['feature_names'] = feature_names
                st.session_state['ml_df'] = ml_df
                
                st.success(f"✅ Model trained! Accuracy: {acc*100:.2f}%")
                
                st.subheader("Model Performance")
                st.pyplot(prediction_model.plot_confusion_matrix(y_test, preds))
                
                importances = prediction_model.get_feature_importance(model, feature_names)
                st.pyplot(prediction_model.plot_feature_importance(importances))
    
    with col2:
        st.subheader("Make Prediction")
        
        if 'model' in st.session_state:
            temp = st.number_input("Temperature (°F)", -20, 120, 70)
            humidity = st.slider("Humidity (%)", 0, 100, 50)
            pressure = st.number_input("Pressure (in)", 28.0, 31.0, 29.92, 0.01)
            visibility = st.number_input("Visibility (mi)", 0.0, 10.0, 10.0, 0.1)
            wind_speed = st.number_input("Wind Speed (mph)", 0.0, 100.0, 5.0, 0.5)
            precipitation = st.number_input("Precipitation (in)", 0.0, 5.0, 0.0, 0.01)
            hour = st.slider("Hour of Day", 0, 23, 12)
            
            weather_options = [col.replace("Weather_Condition_", "") 
                             for col in st.session_state['ml_df'].columns 
                             if col.startswith("Weather_Condition_")]
            weather = st.selectbox("Weather Condition", weather_options)
            
            if st.button("🔮 Predict", type="primary"):
                input_dict = {
                    "Temperature(F)": temp,
                    "Humidity(%)": humidity,
                    "Pressure(in)": pressure,
                    "Visibility(mi)": visibility,
                    "Wind_Speed(mph)": wind_speed,
                    "Precipitation(in)": precipitation,
                    "Hour": hour
                }
                
                for w in weather_options:
                    input_dict[f"Weather_Condition_{w}"] = 1 if w == weather else 0
                
                input_df = pd.DataFrame([input_dict])
                prediction = st.session_state['model'].predict(input_df)[0]
                
                severity_colors = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴"}
                st.markdown(f"### Predicted Severity: {severity_colors.get(prediction, '')} **{prediction}**")
        else:
            st.info("👆 Train a model first to make predictions")

# TAB 5: Hotspot Detection
with tab5:
    st.header("🎯 Accident Hotspot Detection")
    
    st.info("🔍 Using DBSCAN clustering to identify high-risk accident zones")
    
    col1, col2 = st.columns(2)
    with col1:
        eps = st.slider("Clustering Radius (degrees)", 0.1, 2.0, 0.5, 0.1)
        st.caption("Smaller values create tighter clusters")
    with col2:
        min_samples = st.slider("Minimum Accidents per Hotspot", 5, 50, 10, 5)
        st.caption("Higher values identify more significant hotspots")
    
    if st.button("🎯 Detect Hotspots", type="primary"):
        with st.spinner("Analyzing accident patterns..."):
            hotspots, clustered_df = hotspot_detector.detect_hotspots(filtered_df, eps=eps, min_samples=min_samples)
            
            # Store results in session state
            st.session_state['hotspots'] = hotspots
            st.session_state['clustered_df'] = clustered_df
            st.session_state['hotspots_detected'] = True
    
    # Display results if they exist in session state
    if st.session_state.get('hotspots_detected', False):
        hotspots = st.session_state['hotspots']
        clustered_df = st.session_state['clustered_df']
        
        if not hotspots.empty:
            st.success(f"✅ Found {len(hotspots)} accident hotspots!")
            
            # Display top hotspots
            st.subheader("Top 10 Accident Hotspots")
            top_hotspots = hotspots.head(10)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Hotspots", len(hotspots))
            with col2:
                st.metric("Max Accidents in Hotspot", int(hotspots['Accident_Count'].max()))
            with col3:
                st.metric("Avg Severity", f"{hotspots['Avg_Severity'].mean():.2f}")
            
            # Hotspot table
            display_hotspots = top_hotspots.copy()
            display_hotspots['Accident_Count'] = display_hotspots['Accident_Count'].astype(int)
            display_hotspots['Avg_Severity'] = display_hotspots['Avg_Severity'].round(2)
            display_hotspots['Latitude'] = display_hotspots['Latitude'].round(4)
            display_hotspots['Longitude'] = display_hotspots['Longitude'].round(4)
            st.dataframe(display_hotspots, use_container_width=True)
            
            # Create hotspot map
            st.subheader("Hotspot Map")
            center_lat = hotspots['Latitude'].mean()
            center_lng = hotspots['Longitude'].mean()
            
            m = folium.Map(location=[center_lat, center_lng], zoom_start=6, tiles='CartoDB positron')
            
            # Add hotspot circles
            for idx, row in hotspots.iterrows():
                folium.Circle(
                    location=[row['Latitude'], row['Longitude']],
                    radius=row['Accident_Count'] * 100,  # Scale by accident count
                    popup=f"<b>Hotspot {idx}</b><br>Accidents: {int(row['Accident_Count'])}<br>Avg Severity: {row['Avg_Severity']:.2f}",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.4
                ).add_to(m)
                
                # Add marker at center
                folium.Marker(
                    location=[row['Latitude'], row['Longitude']],
                    popup=f"<b>Hotspot {idx}</b><br>Accidents: {int(row['Accident_Count'])}",
                    icon=folium.Icon(color='red', icon='warning-sign')
                ).add_to(m)
            
            st_folium(m, width=1400, height=600, key="hotspot_map")
            
            # Hotspot statistics
            st.subheader("Hotspot Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                # Accidents per hotspot distribution
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(range(len(top_hotspots)), top_hotspots['Accident_Count'], color='coral')
                ax.set_xlabel('Hotspot Rank')
                ax.set_ylabel('Number of Accidents')
                ax.set_title('Top 10 Hotspots by Accident Count', fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Severity distribution in hotspots
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.scatter(top_hotspots['Accident_Count'], top_hotspots['Avg_Severity'], 
                         s=top_hotspots['Accident_Count']*2, alpha=0.6, color='darkred')
                ax.set_xlabel('Accident Count')
                ax.set_ylabel('Average Severity')
                ax.set_title('Hotspot: Accident Count vs Severity', fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
        else:
            st.warning("⚠️ No hotspots detected with current parameters. Try adjusting the settings.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>Traffic Accident Hotspot Analysis System</b></p>
    <p>Data Source: US Accidents Dataset | Powered by Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)