# 🌿 India Air Quality Dashboard
**Python Project Assessment | UPES Dehradun**

## Project Overview
A desktop GUI application that visualizes and analyzes real-time air quality data across Indian cities using multiple Python libraries as required.

## Tech Stack Coverage

| Requirement | Library | How it's used |
|---|---|---|
| GUI (Frontend) | **Tkinter** | Full tabbed dashboard with dropdowns, tables, buttons, live charts |
| Data Handling | **Pandas** | CSV loading, filtering, groupby, pivot tables, cleaning |
| Data Visualization | **Matplotlib** | 5 chart types: bar, horizontal bar, box plot, heatmap, pie |
| Numerical Operations | **NumPy** | Mean, median, std dev, min/max, 75th/95th percentiles |
| Database | **MongoDB** | Full CRUD — Create (bulk insert), Read (query), Update, Delete |

## Dataset
- **Source:** `data.csv` – Real India air quality readings
- **Rows:** ~3,310 records
- **Columns:** country, state, city, station, last_update, latitude, longitude, pollutant_id, pollutant_min, pollutant_max, pollutant_avg
- **Pollutants:** PM2.5, PM10, NO2, SO2, CO, OZONE, NH3
- **Coverage:** 30 states, 248 cities

## Features (5 Tabs)

### 📊 Overview
- KPI cards: total records, states, cities, pollutants
- Bar chart: Top 10 most polluted cities for PM2.5 with safe limit line

### 🔍 Explore
- Filter table by State → City → Pollutant
- Color-coded rows: 🔴 Unhealthy | 🟡 Moderate | 🟢 Good
- Real-time record count

### 📈 Charts (5 types)
1. **State Avg Bar** – average pollutant per state
2. **Box Plot** – distribution of all pollutants
3. **Top Cities Bar** – most polluted cities
4. **State Heatmap** – normalized pollutant matrix
5. **Pie Chart** – pollutant contribution in Delhi

### 🔢 Statistics (NumPy)
- 7 statistical metrics displayed as cards
- Histogram with mean, median, and safe limit lines

### 🗄️ MongoDB CRUD
- **CREATE** – Bulk inserts all 3,310 records on launch
- **READ** – Query by pollutant, shows matching documents
- **UPDATE** – Edit a station's pollutant average
- **DELETE** – Remove all records for a city
- Operation log with timestamps
- Works in offline mode (falls back to CSV) if MongoDB is not running

## Setup & Run

### Requirements
```
pip install pandas matplotlib numpy pymongo
```

### MongoDB (optional but recommended)
1. Install MongoDB Community Edition
2. Start: `mongod` or start MongoDB service
3. No config needed — app auto-connects to `localhost:27017`

> **Note:** If MongoDB is not running, the app still works fully except CRUD tab shows CSV fallback data.

### Run
```bash
cd AirQualityDashboard
python main.py
```

## File Structure
```
AirQualityDashboard/
├── main.py       ← Main application (single file)
├── data.csv      ← Dataset
└── README.md     ← This file
```
