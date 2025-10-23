import pandas as pd
import os
import json

def load_data(file_path):
    """
    Load CSV, select relevant columns, drop nulls and duplicates.
    """
    df = pd.read_csv(file_path)
    cols = ["ID","Severity","Start_Time","Start_Lat","Start_Lng","City","State"]
    df = df[cols].dropna(subset=["Start_Lat","Start_Lng","City"]).drop_duplicates()
    return df

def create_map(df, limit=1000, output_file="accident_map.html"):
    """
    Generate interactive HTML map with accidents, source-destination selection,
    buffer distance, and route statistics.
    """
    accident_data = df.head(limit).to_dict(orient="records")
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>US Accidents Route Planner</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        #map {{ height: 100vh; }}
        .route-control{{
            position: fixed; top:10px; right:10px; background:white; padding:15px;
            border-radius:5px; box-shadow:0 2px 5px rgba(0,0,0,0.3); z-index:1000; max-width:300px;
        }}
        .route-control h3{{margin-top:0; color:#333;}}
        .route-control button{{background:#ff4444;color:white;border:none;padding:8px 15px;border-radius:3px;cursor:pointer;margin-top:10px;}}
        .route-control button:hover{{background:#cc0000;}}
        .slider-container{{margin-top:10px;}}
        .slider-container input{{width:100%;}}
        #route-stats{{margin-top:10px;padding:10px;background:#f0f0f0;border-radius:3px;font-size:14px;}}
    </style>
</head>
<body>
<div id="map"></div>
<div class="route-control">
    <h3>🗺️ Route Planner</h3>
    <p>Click map to set <span style="color:green">SOURCE</span> and <span style="color:red">DESTINATION</span></p>
    <div class="slider-container">
        <label><b>Buffer Distance:</b> <span id="buffer-value">5 km</span></label>
        <input type="range" min="1" max="50" value="5" step="1" id="bufferSlider">
    </div>
    <div id="route-stats">Click on map to set source and destination</div>
    <button id="clearRoute">🗑️ Clear Route</button>
</div>

<script>
var map = L.map('map').setView([39.5, -98.35], 4);

// Add OSM layer
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19
}}).addTo(map);

// Accident data
var accidentData = {json.dumps(accident_data)};
var accidentMarkers = [];
accidentData.forEach(function(accident){{
    var m = L.circleMarker([accident.Start_Lat, accident.Start_Lng], {{
        radius:2, color:'red', fillColor:'red', fillOpacity:0.5
    }}).bindPopup(
        '<b>Severity:</b>'+accident.Severity+'<br>' +
        '<b>Location:</b>'+accident.City+', '+accident.State+'<br>' +
        '<b>Time:</b>'+accident.Start_Time
    ).addTo(map);
    accidentMarkers.push(m);
}});

// Route variables
var sourceMarker=null, destMarker=null, routeLine=null, highlightedMarkers=[], bufferDistance=5;

// Distance functions
function getDistance(lat1, lng1, lat2, lng2){{
    var R=6371;
    var dLat=(lat2-lat1)*Math.PI/180;
    var dLng=(lng2-lng1)*Math.PI/180;
    var a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLng/2)**2;
    return R*2*Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}}

function pointToLineDistance(px, py, x1, y1, x2, y2){{
    var A=px-x1, B=py-y1, C=x2-x1, D=y2-y1;
    var dot=A*C+B*D, len_sq=C*C+D*D, param=(len_sq!=0)?dot/len_sq:-1;
    var xx=(param<0)?x1:(param>1)?x2:x1+param*C;
    var yy=(param<0)?y1:(param>1)?y2:y1+param*D;
    return getDistance(px, py, xx, yy);
}}

function highlightRouteAccidents(){{
    if(!sourceMarker||!destMarker) return;
    accidentMarkers.forEach(function(m){{ m.setStyle({{radius:2,color:'red',fillColor:'red',fillOpacity:0.5}}); }} );
    highlightedMarkers=[];
    var s=sourceMarker.getLatLng(), d=destMarker.getLatLng();
    accidentMarkers.forEach(function(m){{
        var p=m.getLatLng();
        if(pointToLineDistance(p.lat,p.lng,s.lat,s.lng,d.lat,d.lng)<=bufferDistance){{
            m.setStyle({{radius:5,color:'yellow',fillColor:'orange',fillOpacity:0.8}});
            highlightedMarkers.push(m);
        }}
    }} );
    updateStats();
}}

function updateStats(){{
    var statsDiv=document.getElementById('route-stats');
    if(sourceMarker && destMarker){{
        var dist=getDistance(sourceMarker.getLatLng().lat,sourceMarker.getLatLng().lng,destMarker.getLatLng().lat,destMarker.getLatLng().lng);
        statsDiv.innerHTML='<b>Route Distance:</b>'+dist.toFixed(2)+' km<br>' +
                           '<b>Accidents on Route:</b>'+highlightedMarkers.length+'<br>' +
                           '<b>Buffer Zone:</b> ±'+bufferDistance+' km';
    }} else {{
        statsDiv.innerHTML='Click on map to set source and destination';
    }}
}}

// Map click to set markers
map.on('click', function(e){{
    if(!sourceMarker){{
        sourceMarker=L.marker(e.latlng, {{icon:L.icon({{
            iconUrl:'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
            shadowUrl:'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
            iconSize:[25,41],iconAnchor:[12,41],popupAnchor:[1,-34],shadowSize:[41,41]
        }})}}).addTo(map).bindPopup('<b>SOURCE</b>').openPopup();
    }} else if(!destMarker){{
        destMarker=L.marker(e.latlng, {{icon:L.icon({{
            iconUrl:'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
            shadowUrl:'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
            iconSize:[25,41],iconAnchor:[12,41],popupAnchor:[1,-34],shadowSize:[41,41]
        }})}}).addTo(map).bindPopup('<b>DESTINATION</b>').openPopup();
        routeLine=L.polyline([sourceMarker.getLatLng(),destMarker.getLatLng()],{{color:'blue',weight:3,opacity:0.7}}).addTo(map);
        highlightRouteAccidents();
    }}
}});

// Clear Route
document.getElementById('clearRoute').onclick=function(){{
    if(sourceMarker) map.removeLayer(sourceMarker); sourceMarker=null;
    if(destMarker) map.removeLayer(destMarker); destMarker=null;
    if(routeLine) map.removeLayer(routeLine); routeLine=null;
    accidentMarkers.forEach(function(m){{ m.setStyle({{radius:2,color:'red',fillColor:'red',fillOpacity:0.5}}); }} );
    highlightedMarkers=[]; updateStats();
}}

// Buffer slider
document.getElementById('bufferSlider').oninput=function(e){{
    bufferDistance=parseFloat(e.target.value);
    document.getElementById('buffer-value').innerText=bufferDistance+' km';
    highlightRouteAccidents();
}}
</script>
</body>
</html>
"""
    # Save HTML
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Map saved as {output_file}")
