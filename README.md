# Traffic Accident Hotspot Identification and Analysis 🚗📊

A comprehensive Big Data Analytics project for identifying, analyzing, and visualizing traffic accident hotspots across the United States using advanced data processing and machine learning techniques.

## 🎯 Project Overview

This project leverages big data technologies to analyze millions of traffic accident records, identify high-risk zones, predict accident-prone areas, and provide interactive visualizations for route planning and safety analysis.

## ✨ Features

### Current Implementation
- 🗺️ **Interactive Route Planner**: Google Maps-style interface for route selection
- 📍 **Accident Visualization**: Real-time filtering of accidents along routes
- 🎯 **Buffer Zone Analysis**: Adjustable distance-based accident detection
- 📊 **Route Statistics**: Accident count and severity distribution

## 🛠️ Technologies Used

### Big Data & Analytics
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Statistical visualizations

### Visualization
- **Leaflet.js**: Interactive maps
- **Plotly**: Advanced data visualizations
- **D3.js**: Custom data-driven graphics

## 📁 Project Structure

```
traffic-accident-hotspot/
├── data/
│   ├── US_Accidents_March23.csv             # download from link
├── scripts/
│   ├── map_display.py                     
├── accident_map.html                        # after running main.py                
├── requirements.txt                       
├── .gitattributes                         
├── .gitignore
├── main.py                                  # Main application
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended

### Installation

#### 1. Clone Repository
```bash
git clone https://github.com/sneh-a-15/traffic-accident-hotspot.git
cd traffic-accident-hotspot

```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run Application
```bash
python main.py
```

#### 5. Open Visualizations
- Interactive Map: Open `accident_map.html`

## 📊 Dataset Information

**Source**: [US Accidents (2016-2023) - Kaggle](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)      # Download dataset from this link

**Size**: ~7 million records, ~3GB

**Key Attributes**:
- `ID`: Unique accident identifier
- `Severity`: Accident severity (1-4 scale)
- `Start_Time`, `End_Time`: Temporal information
- `Start_Lat`, `Start_Lng`: Geospatial coordinates
- `City`, `State`, `County`: Location details
- `Temperature`, `Humidity`, `Visibility`: Weather conditions
- `Weather_Condition`: Weather description
- `Road_Features`: Traffic signals, crossings, junctions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request




