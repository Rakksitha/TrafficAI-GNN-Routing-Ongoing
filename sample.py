import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import requests
import plotly.express as px
import numpy as np

# Streamlit Page Configuration
st.set_page_config(page_title="AI Traffic Optimizer", page_icon="🚦", layout="wide")
st.title("🚀 AI-Optimized Traffic Flow & Route Planning")

# Predefined Route Options (3-5 choices)
routes = {
    "Route 1: City Center to University": [(11.6632, 78.1462), (11.6612, 78.1510)],
    "Route 2: Tech Park to Railway Station": [(11.6711, 78.1605), (11.6523, 78.1407)],
    "Route 3: Mall to Airport": [(11.6750, 78.1482), (11.6895, 78.1753)],
}

# Sidebar Selection for Route
st.sidebar.header("📍 Choose a Route")
route_choice = st.sidebar.selectbox("Select a Route", list(routes.keys()))
start_coords, dest_coords = routes[route_choice]

# Sidebar Selection for Route Priority
st.sidebar.subheader("🚦 Select Route Priority")
priority = st.sidebar.radio("Choose Priority:", ["Red (Fastest)", "Yellow (Balanced)", "Green (Safest)"])
priority = priority.split(" ")[0].lower()  # Extracts "red", "yellow", or "green"

# Route Optimization
st.header("🛣️ AI-Optimized Route Suggestion")
if st.button("Get Optimized Route 🚗"):
    start_x, start_y = start_coords
    dest_x, dest_y = dest_coords
    
    try:
        response = requests.get(
            f"http://127.0.0.1:5000/get_route?start_x={start_x}&start_y={start_y}&dest_x={dest_x}&dest_y={dest_y}&priority={priority}",
            timeout=5
        )
        response.raise_for_status()
        route_data = response.json().get("route", [])
        
        if route_data:
            st.success(f"🚦 Optimized Route Found Based on {priority.capitalize()} Priority!")

            # Create a Map with the Optimized Route
            route_map = folium.Map(location=[start_x, start_y], zoom_start=14)
            folium.Marker([start_x, start_y], popup="Start", icon=folium.Icon(color='green')).add_to(route_map)
            folium.Marker([dest_x, dest_y], popup="Destination", icon=folium.Icon(color='red')).add_to(route_map)
            
            # Define route colors based on priority
            route_color = {"red": "red", "yellow": "orange", "green": "blue"}
            folium.PolyLine(route_data, color=route_color[priority], weight=4).add_to(route_map)

            folium_static(route_map)
        else:
            st.warning("⚠️ No route data received from the backend. Try again!")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {e}")

# Dummy Data for Traffic Flow Over Time
np.random.seed(42)
time_series = pd.date_range(start="2025-03-01", periods=10, freq="D")
traffic_values = np.random.randint(20, 100, size=10)

# Creating Line Chart for Traffic Flow
fig_line = px.line(
    x=time_series, 
    y=traffic_values, 
    title="🚦 Traffic Flow Over Time",
    labels={"x": "Time", "y": "Traffic Intensity"},
    markers=True
)

# Dummy Data for Traffic Congestion Heatmap
heatmap_data = pd.DataFrame(
    np.random.rand(10, 10) * 100,
    columns=[f"Junction {i+1}" for i in range(10)]
)

# Creating Heatmap for Traffic Congestion
fig_heatmap = px.imshow(
    heatmap_data, 
    labels={"x": "Junctions", "y": "Time", "color": "Congestion Level"},
    title="🗺️ Traffic Congestion Heatmap",
    color_continuous_scale="reds"
)

# Display Charts in a Single Row
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.plotly_chart(fig_heatmap, use_container_width=True)

# Footer
st.caption("🚀 AI-Powered Traffic Flow Optimization | Hackathon Prototype")
