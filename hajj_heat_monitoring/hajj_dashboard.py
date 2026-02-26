"""
Hajj Heat Risk Monitoring Dashboard
A Streamlit application for real-time visualization of crowd-modified wet-bulb heat risk
Using actual data from your pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pytz
from pathlib import Path

from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import Polygon, MultiPoint, LineString, MultiLineString
from shapely.ops import unary_union, polygonize

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Hajj Heat Risk Monitor",
    page_icon="🕋",
    layout="wide",
    initial_sidebar_state="expanded"
)

DASHBOARD_SVG = """
<svg fill="#000000" height="100px" width="100px" version="1.1" id="Layer_1" 
xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
viewBox="0 0 512 512" xml:space="preserve"><g id="SVGRepo_bgCarrier" stroke-width="0"></g>
<g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <g> <g> 
<path d="M486.881,59.858H25.119C11.268,59.858,0,71.126,0,84.977v273.637c0,13.851,11.268,25.119,25.119,25.119h171.557v52.376 
h-26.188c-4.427,0-8.017,3.589-8.017,8.017c0,4.427,3.589,8.017,8.017,8.017h171.023c4.427,0,8.017-3.589,8.017-8.017 
c0-4.427-3.589-8.017-8.017-8.017h-26.188v-52.376h171.557c13.851,0,25.119-11.268,25.119-25.119V84.977 
C512,71.126,500.732,59.858,486.881,59.858z M299.29,436.109h-86.58v-52.376h86.58V436.109z M495.967,358.614 
c0,5.01-4.076,9.086-9.086,9.086H25.119c-5.01,0-9.086-4.076-9.086-9.086v-9.086h479.933V358.614z M495.967,333.495H16.033V84.977 
c0-5.01,4.076-9.086,9.086-9.086h461.762c5.01,0,9.086,4.076,9.086,9.086V333.495z"></path> </g> </g> <g> <g> 
<path d="M144.835,102.614c-56.287,0-102.079,45.792-102.079,102.079s45.792,102.079,102.079,102.079 
s102.079-45.792,102.079-102.079S201.122,102.614,144.835,102.614z M144.835,118.647c20.842,0,39.974,7.453,54.88,19.828 
l-25.12,25.12c-8.369-6.078-18.65-9.675-29.761-9.675c-25.268,0-46.278,18.556-50.133,42.756H59.176 
C63.234,152.977,100.093,118.647,144.835,118.647z M66.073,239.281c-3.644-8.265-6.026-17.2-6.896-26.572h35.526 
c0.513,3.217,1.324,6.336,2.414,9.325L66.073,239.281z M136.818,290.352c-26.099-2.423-48.848-16.544-62.945-37.065l31.06-17.255 
c7.736,9.827,19.006,16.742,31.885,18.794V290.352z M110.096,204.693c0-19.155,15.584-34.739,34.739-34.739 
c19.155,0,34.739,15.584,34.739,34.739s-15.584,34.739-34.739,34.739C125.68,239.432,110.096,223.848,110.096,204.693z 
M152.852,290.352v-35.526c24.2-3.855,42.756-24.866,42.756-50.133c0-11.111-3.597-21.392-9.675-29.761l25.12-25.12 
c12.375,14.907,19.828,34.039,19.828,54.881C230.881,249.435,196.551,286.295,152.852,290.352z"></path> </g> </g> <g> <g> 
<path d="M461.228,205.228H290.205c-4.427,0-8.017,3.589-8.017,8.017s3.589,8.017,8.017,8.017h171.023 
c4.427,0,8.017-3.589,8.017-8.017S465.655,205.228,461.228,205.228z"></path> </g> </g> <g> <g> 
<path d="M315.858,247.983h-25.653c-4.427,0-8.017,3.589-8.017,8.017s3.589,8.017,8.017,8.017h25.653 
c4.427,0,8.017-3.589,8.017-8.017S320.285,247.983,315.858,247.983z"></path> </g> </g> <g> <g> 
<path d="M418.472,247.983h-68.409c-4.427,0-8.017,3.589-8.017,8.017s3.589,8.017,8.017,8.017h68.409 
c4.427,0,8.017-3.589,8.017-8.017S422.899,247.983,418.472,247.983z"></path> </g> </g> <g> <g> 
<path d="M444.125,282.188h-94.063c-4.427,0-8.017,3.589-8.017,8.017s3.589,8.017,8.017,8.017h94.063 
c4.427,0,8.017-3.589,8.017-8.017S448.553,282.188,444.125,282.188z"></path> </g> </g> <g> <g> 
<path d="M315.858,282.188h-25.653c-4.427,0-8.017,3.589-8.017,8.017s3.589,8.017,8.017,8.017h25.653 
c4.427,0,8.017-3.589,8.017-8.017S320.285,282.188,315.858,282.188z"></path> </g> </g> <g> <g> 
<path d="M290.205,128.267c-4.427,0-8.017,3.589-8.017,8.017v42.756c0,4.427,3.589,8.017,8.017,8.017s8.017-3.589,8.017-8.017 
v-42.756C298.221,131.857,294.632,128.267,290.205,128.267z"></path> </g> </g> <g> <g> 
<path d="M392.818,128.267c-4.427,0-8.017,3.589-8.017,8.017v42.756c0,4.427,3.589,8.017,8.017,8.017 
c4.427,0,8.017-3.589,8.017-8.017v-42.756C400.835,131.857,397.246,128.267,392.818,128.267z"></path> </g> </g> <g> <g> 
<path d="M358.614,153.921c-4.427,0-8.017,3.589-8.017,8.017v17.102c0,4.427,3.589,8.017,8.017,8.017s8.017-3.589,8.017-8.017 
v-17.102C366.63,157.51,363.041,153.921,358.614,153.921z"></path> </g> </g> <g> <g> 
<path d="M427.023,153.921c-4.427,0-8.017,3.589-8.017,8.017v17.102c0,4.427,3.589,8.017,8.017,8.017 
c4.427,0,8.017-3.589,8.017-8.017v-17.102C435.04,157.51,431.45,153.921,427.023,153.921z"></path> </g> </g> <g> <g> 
<path d="M324.409,111.165c-4.427,0-8.017,3.589-8.017,8.017v59.858c0,4.427,3.589,8.017,8.017,8.017s8.017-3.589,8.017-8.017 
v-59.858C332.426,114.754,328.837,111.165,324.409,111.165z"></path> </g> </g> <g> <g> 
<path d="M461.228,111.165c-4.427,0-8.017,3.589-8.017,8.017v59.858c0,4.427,3.589,8.017,8.017,8.017 
c4.427,0,8.017-3.589,8.017-8.017v-59.858C469.244,114.754,465.655,111.165,461.228,111.165z"></path> </g> </g> </g>
</svg>
""".strip()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .green { background-color: #d4edda; color: #155724; border-left: 5px solid #28a745; }
    .yellow { background-color: #fff3cd; color: #856404; border-left: 5px solid #ffc107; }
    .orange { background-color: #ffe5b4; color: #854d0e; border-left: 5px solid #fd7e14; }
    .red { background-color: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }
    .black { background-color: #343a40; color: white; border-left: 5px solid #000000; }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }

    /* ---------- SIDEBAR ---------- */

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    #stDecoration {display:none;}

    /* Let Streamlit theme handle main colors; just nudge neutrals */
    :root {
        --primary-color: #d4d4d8 !important;              /* neutral gray */
        --secondary-background-color: #f5f5f7 !important; /* cards / sidebar */
        --text-color: #111827 !important;                 /* near black */
    }

    section[data-testid="stSidebar"] {
        background: #EDEDF2 !important;
        border-right: 1px solid #d4d4d8 !important;
    }

    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 260px;
        max-width: 260px;
        width: 260px;
        box-shadow: 0 0 12px rgba(0, 0, 0, 0.08);
        transition: all 0.25s ease-in-out;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        transition: all 0.25s ease-in-out;
        overflow: hidden;
    }

</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Site coordinates (from your code)
SITES = {
    "HARAM": {"lat": 21.4225, "lon": 39.8262, "color": "#FF6B6B"},
    "MINA": {"lat": 21.4133, "lon": 39.8947, "color": "#4ECDC4"},
    "ARAFAT": {"lat": 21.3550, "lon": 39.9840, "color": "#FFD166"},
    "MUZDALIFAH": {"lat": 21.3860, "lon": 39.9400, "color": "#6C5B7B"}
}

# Alert thresholds (from your code)
ALERT_THRESHOLDS = [
    (0, 24, "GREEN", "Normal operations", "#28a745"),
    (24, 26, "YELLOW", "Extra water stations; increased medical presence", "#ffc107"),
    (26, 28, "ORANGE", "Mandatory rest periods; shade structures required", "#fd7e14"),
    (28, 30, "RED", "Suspend outdoor rituals for elderly/medically vulnerable", "#dc3545"),
    (30, 100, "BLACK", "Complete suspension of outdoor activities", "#000000")
]

ZONE_COUNTS = {"ARAFAT": 12, "MUZDALIFAH": 8, "MINA": 14, "HARAM": 4}
DATA_DIR = Path("data")
ZONES_DIR = DATA_DIR / "zones"
ZONES_DIR.mkdir(parents=True, exist_ok=True)

def get_alert_info(temp):
    """Return alert color and message based on temperature"""
    if pd.isna(temp):
        return "UNKNOWN", "No data"
    for low, high, color, msg, _ in ALERT_THRESHOLDS:
        if low <= temp < high:
            return color, msg
    return "UNKNOWN", "Unknown conditions"

def get_alert_color(temp):
    """Return hex color for alert level"""
    if pd.isna(temp):
        return "#808080"
    for low, high, _, _, hex_color in ALERT_THRESHOLDS:
        if low <= temp < high:
            return hex_color
    return "#808080"

def get_alert_level(temp):
    """Return alert level as string"""
    if pd.isna(temp):
        return "UNKNOWN"
    for low, high, color, _, _ in ALERT_THRESHOLDS:
        if low <= temp < high:
            return color
    return "UNKNOWN"

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_cell_risk_data(year):
    fp = Path(f"data/hourly_risk_cells_{year}_wetbulb.csv")
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def load_site_summary_data(years=[2021, 2022, 2023, 2024, 2025]):
    """
    Load your hourly risk site summary data from CSV files
    """
    all_data = []
    data_dir = Path("data/")
    
    # Create directory if it doesn't exist (for demo purposes)
    if not data_dir.exists():
        st.warning(f"Data directory not found: {data_dir}")
        return pd.DataFrame()
    
    for year in years:
        file_path = data_dir / f"hourly_risk_site_summary_{year}_wetbulb.csv"
        
        if not file_path.exists():
            st.warning(f"File not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = year
        
        # Calculate hajj day if not present
        if 'hajj_day' not in df.columns:
            # Hajj start dates (8 Dhul-Hijjah) for each year
            hajj_starts = {
                2021: '2021-07-18',
                2022: '2022-07-07',
                2023: '2023-06-26',
                2024: '2024-06-14',
                2025: '2025-06-04'
            }
            start_date = pd.Timestamp(hajj_starts[year])
            df['hajj_day'] = (df['timestamp'] - start_date).dt.days
        
        # Extract hour
        df['hour'] = df['timestamp'].dt.hour
        
        all_data.append(df)
    
    if not all_data:
        st.error("No data files found. Please check the file paths.")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    return combined

def load_amplification_map():
    """
    Load your satellite_with_amp01.csv data (for research view only)
    """
    file_path = Path("data//satellite_with_amp01.csv")
    
    if not file_path.exists():
        st.error(f"Amplification map file not found: {file_path}")
        return pd.DataFrame(columns=['lon', 'lat', 'amp01', 'site'])
    
    df = pd.read_csv(file_path)
    return df

def load_year_data(year):
    """Load data for a specific year only"""
    df = load_site_summary_data(years=[year])
    return df

# =============================================================================
# CONCAVE HULL (ALPHA SHAPE)
# =============================================================================

def alpha_shape_boundary(points_lonlat: np.ndarray, alpha: float = 150.0, buffer_deg: float = 0.00035):
    """
    Concave hull (alpha shape) from lon/lat points using Delaunay triangulation.
    Larger alpha => tighter hull.
    """
    pts = np.asarray(points_lonlat, dtype=float)
    if len(pts) < 4:
        return MultiPoint(pts).convex_hull.buffer(buffer_deg)

    tri = Delaunay(pts)
    triangles = pts[tri.simplices]

    def circumradius(a, b, c):
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)
        s = (ab + bc + ca) / 2.0
        area2 = s * (s - ab) * (s - bc) * (s - ca)
        if area2 <= 0:
            return np.inf
        area = np.sqrt(area2)
        return (ab * bc * ca) / (4.0 * area)

    edges = []
    for a, b, c in triangles:
        r = circumradius(a, b, c)
        if r < (1.0 / alpha):
            edges.append(LineString([a, b]))
            edges.append(LineString([b, c]))
            edges.append(LineString([c, a]))

    if not edges:
        return MultiPoint(pts).convex_hull.buffer(buffer_deg)

    mls = unary_union(edges)
    polys = list(polygonize(mls))
    if not polys:
        return MultiPoint(pts).convex_hull.buffer(buffer_deg)

    boundary = unary_union(polys).buffer(0)

    if boundary.geom_type == "MultiPolygon":
        boundary = max(list(boundary.geoms), key=lambda p: p.area)

    return boundary.buffer(buffer_deg)

# =============================================================================
# VORONOI FINITE POLYGONS
# =============================================================================

def voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    """
    Convert infinite Voronoi regions to finite polygons.
    Returns (regions, vertices) aligned to vor.points order.
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]

        if region and all(v >= 0 for v in region):
            new_regions.append(region)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [v for v in region if v >= 0] if region else []

        for p2, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            norm = np.linalg.norm(t)
            if norm == 0:
                continue
            t = t / norm
            n = np.array([-t[1], t[0]])

            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            direction = np.sign(np.dot(midpoint - center, n)) * n

            finite_v = vor.vertices[v1] if v1 >= 0 else vor.vertices[v2]
            far_point = finite_v + direction * radius

            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        if not new_region:
            new_regions.append([])
            continue

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]

        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

# =============================================================================
# BUILD ZONES
# =============================================================================

def stable_zone_labels_from_kmeans(df_site_unique: pd.DataFrame, k: int) -> np.ndarray:
    """
    KMeans labels remapped into stable zone indices by sorting centroids:
    south->north then west->east. Gives consistent zone numbering per run.
    (Not persisted across runs, but stable for the current dataset.)
    """
    X = df_site_unique[["lon", "lat"]].to_numpy()
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X)

    centers = pd.DataFrame(km.cluster_centers_, columns=["lon_c", "lat_c"])
    centers["cluster"] = np.arange(k)
    centers = centers.sort_values(["lat_c", "lon_c"], ascending=[True, True]).reset_index(drop=True)

    remap = {int(row["cluster"]): int(i) for i, row in centers.iterrows()}
    return np.array([remap[int(c)] for c in labels], dtype=int)

def compute_zones_and_polygons_for_time(
    cell_df: pd.DataFrame,
    selected_time: pd.Timestamp,
    zone_counts: dict,
    alpha_by_site: dict,
):
    """
    NON-PERSISTENT:
    For the selected_time, build zones per site using:
      - stable KMeans labels
      - centroid Voronoi polygons
      - clipped to concave alpha-shape boundary
    Returns zones_meta_df (polygons) and zone_stats_df (mean/max/%>= etc.)
    """

    time_cells = cell_df[cell_df["timestamp"] == selected_time].copy()
    if time_cells.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ensure numeric
    time_cells["lon"] = pd.to_numeric(time_cells["lon"], errors="coerce")
    time_cells["lat"] = pd.to_numeric(time_cells["lat"], errors="coerce")
    time_cells["twb_eff"] = pd.to_numeric(time_cells.get("twb_eff", np.nan), errors="coerce")
    time_cells = time_cells.dropna(subset=["lon", "lat", "twb_eff", "site"])

    zones_meta_rows = []
    zone_stats_rows = []

    for site in sorted(time_cells["site"].unique()):
        df_s = time_cells[time_cells["site"] == site].copy()
        if len(df_s) < 5:
            continue

        k_target = int(zone_counts.get(site, 1))
        k = min(k_target, max(1, len(df_s)))

        if k <= 1:
            stable_labels = np.zeros(len(df_s), dtype=int)
        else:
            stable_labels = stable_zone_labels_from_kmeans(df_s[["lon", "lat"]].assign(site=site), k)

        df_s["zone_idx"] = stable_labels
        df_s["zone_id"] = df_s["zone_idx"].apply(lambda i: f"{site[0]}{int(i)+1}")

        # centers in zone_idx order
        centers = (
            df_s.groupby("zone_idx")[["lon", "lat"]]
            .mean()
            .reset_index()
            .sort_values("zone_idx")
        )
        centers_lonlat = centers[["lon", "lat"]].to_numpy()

        site_pts_lonlat = df_s[["lon", "lat"]].to_numpy()

        # concave boundary
        a = float(alpha_by_site.get(site, 150.0))
        boundary = alpha_shape_boundary(site_pts_lonlat, alpha=a, buffer_deg=0.00035)

        # Voronoi polygons clipped
        vor = Voronoi(centers_lonlat)
        regions, vertices = voronoi_finite_polygons_2d(vor, radius=1.0)

        # stats per zone
        def pct_ge(arr, thr):
            arr = pd.to_numeric(arr, errors="coerce")
            arr = arr[~pd.isna(arr)]
            if len(arr) == 0:
                return np.nan
            return float((arr >= thr).mean() * 100.0)

        by_zone = df_s.groupby("zone_idx")

        for zone_idx, region in enumerate(regions):
            zone_id = f"{site[0]}{zone_idx+1}"

            poly = Polygon(vertices[region]).intersection(boundary).buffer(0)
            if poly.is_empty:
                continue
            if poly.geom_type == "MultiPolygon":
                poly = max(list(poly.geoms), key=lambda p: p.area)

            x, y = poly.exterior.coords.xy
            poly_lon = list(x)
            poly_lat = list(y)

            dz = by_zone.get_group(zone_idx) if zone_idx in by_zone.groups else df_s.iloc[0:0]
            vals = dz["twb_eff"].values

            mean_v = float(np.nanmean(vals)) if len(vals) else np.nan
            max_v = float(np.nanmax(vals)) if len(vals) else np.nan
            n = int(np.sum(~np.isnan(vals))) if len(vals) else 0
            p28 = pct_ge(vals, 28)
            p30 = pct_ge(vals, 30)

            zone_stats_rows.append(
                {
                    "site": site,
                    "zone_id": zone_id,
                    "mean": mean_v,
                    "max": max_v,
                    "p_ge_28": p28,
                    "p_ge_30": p30,
                    "n": n,
                    "alert": get_alert_level(mean_v),
                    "color": get_alert_color(mean_v),
                }
            )

            zones_meta_rows.append(
                {
                    "site": site,
                    "zone_id": zone_id,
                    "polygon_lon": poly_lon,
                    "polygon_lat": poly_lat,
                }
            )

    zones_meta_df = pd.DataFrame(zones_meta_rows)
    zone_stats_df = pd.DataFrame(zone_stats_rows)
    return zones_meta_df, zone_stats_df


def build_folium_zones_map(zones_meta_df, zone_stats_df, selected_time):
    center_lat = 21.389
    center_lon = 39.899

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Tile layers (your style)
    folium.TileLayer("CartoDB positron", name="Light Map", overlay=False, control=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Map", overlay=False, control=True).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)",
        name="Topographic",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    # Draw polygons
    z = zones_meta_df.merge(zone_stats_df, on=["site", "zone_id"], how="left")

    for _, r in z.iterrows():
        locations = list(zip(r["polygon_lat"], r["polygon_lon"]))
        color = r.get("color", "#808080")

        mean_v = r.get("mean", np.nan)
        max_v = r.get("max", np.nan)
        p28 = r.get("p_ge_28", np.nan)
        p30 = r.get("p_ge_30", np.nan)
        n = r.get("n", 0)
        alert = r.get("alert", "UNKNOWN")

        popup_html = f"""
        <div style="min-width: 260px; font-family: Arial, sans-serif;">
            <h4 style="margin:0; color:{color};">{r['site']} - {r['zone_id']}</h4>
            <hr style="margin:6px 0;">
            <p style="margin:2px 0;"><b>Mean twb_eff:</b> {mean_v:.2f} °C</p>
            <p style="margin:2px 0;"><b>Max twb_eff:</b> {max_v:.2f} °C</p>
            <p style="margin:2px 0;"><b>% ≥ 28°C:</b> {p28:.1f}%</p>
            <p style="margin:2px 0;"><b>% ≥ 30°C:</b> {p30:.1f}%</p>
            <p style="margin:2px 0;"><b>Cells:</b> {int(n)}</p>
            <p style="margin:6px 0 0 0;"><b>Alert:</b> <span style="color:{color};">{alert}</span></p>
            <p style="margin:8px 0 0 0; font-size: 0.9em; color:#666;">
                {selected_time.strftime('%Y-%m-%d %H:%M')}
            </p>
        </div>
        """

        folium.Polygon(
            locations=locations,
            color=color,
            weight=3,
            fill=True,
            fill_color=color,
            fill_opacity=0.30,
            popup=folium.Popup(popup_html, max_width=380),
            tooltip=f"{r['site']} {r['zone_id']} | mean={mean_v:.1f}°C | {alert}",
        ).add_to(m)

    # Landmarks (your list)
    landmarks = [
        {"name": "Kaaba", "coords": [21.4225, 39.8262], "color": "red", "icon": "star"},
        {"name": "Jabal al-Rahmah", "coords": [21.3548, 39.9837], "color": "green", "icon": "mountain"},
        {"name": "Jamarat Bridge", "coords": [21.4042, 39.8967], "color": "purple", "icon": "tower"},
        {"name": "Masjid Nimrah", "coords": [21.3522, 39.9625], "color": "green", "icon": "mosque"},
        {"name": "Masjid al-Khayf", "coords": [21.4117, 39.8933], "color": "orange", "icon": "mosque"},
        {"name": "Masjid al-Mash'ar al-Haram", "coords": [21.3883, 39.9183], "color": "purple", "icon": "mosque"},
    ]
    for lm in landmarks:
        folium.Marker(
            location=lm["coords"],
            popup=f"<b>{lm['name']}</b>",
            tooltip=lm["name"],
            icon=folium.Icon(color=lm["color"], icon=lm["icon"], prefix="fa"),
        ).add_to(m)

    # Tools
    plugins.MeasureControl(
        position="topleft",
        primary_length_unit="kilometers",
        secondary_length_unit="miles",
        primary_area_unit="sq. kilometers",
        secondary_area_unit="sq. miles",
    ).add_to(m)
    plugins.Fullscreen().add_to(m)
    folium.LayerControl().add_to(m)

    # title_html = f"""
    # <div style="position: fixed; top: 10px; left: 50px; z-index: 1000; background-color: white;
    #             padding: 10px; border-radius: 5px; border: 2px solid grey; opacity: 0.9;">
    #     <h4 style="margin: 0;">Operational Zones (Non-Persistent)</h4>
    #     <p style="margin: 5px 0 0 0; font-size: 12px;">
    #         Time: <b>{selected_time.strftime("%Y-%m-%d %H:%M")}</b><br>
    #         Concave boundary + Voronoi tiling
    #     </p>
    # </div>
    # """
    # m.get_root().html.add_child(folium.Element(title_html))

    return m

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

#st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2327/2327991.png", width=100)
# Create columns in sidebar to center the image
col1, col2, col3 = st.sidebar.columns([1, 2, 1])
with col2:
    st.image("images/data-analysis.png", width=100)
    #st.sidebar.title("🕋 Hajj Heat Risk Monitor")
    #st.markdown(DASHBOARD_SVG, unsafe_allow_html=True)

# Data selection
st.sidebar.header("📅 Data Selection")
selected_year = st.sidebar.selectbox(
    "Select Year",
    options=[2021, 2022, 2023, 2024, 2025],
    index=4  # Default to 2024
)

# Load data
with st.spinner("Loading data..."):
    df_year = load_year_data(selected_year)
    amp_df = load_amplification_map()
    
    # If year data is empty, load all as fallback
    if df_year.empty:
        st.warning(f"No data for {selected_year}, loading all years")
        df_all = load_site_summary_data()
    else:
        df_all = df_year

# Check if data loaded successfully
if df_all.empty:
    st.error("No data loaded. Please check your data files.")
    st.stop()

# Get unique timestamps for this year
timestamps = sorted(df_all['timestamp'].unique())

# Date/time selector
st.sidebar.header("⏰ Time Selection")

# Date selector
unique_dates = sorted(df_all['timestamp'].dt.date.unique())
selected_date = st.sidebar.selectbox(
    "Select Date",
    options=unique_dates,
    format_func=lambda x: x.strftime("%Y-%m-%d"),
    index=len(unique_dates)//2 if unique_dates else 0
)

# Hour selector for selected date
date_mask = df_all['timestamp'].dt.date == selected_date
hours_on_date = sorted(df_all[date_mask]['hour'].unique())
selected_hour = st.sidebar.selectbox(
    "Select Hour",
    options=hours_on_date,
    format_func=lambda x: f"{x:02d}:00"
)

selected_time = pd.Timestamp(f"{selected_date} {selected_hour}:00:00")

# Site selector
st.sidebar.header("📍 Site Selection")
sites_in_data = sorted(df_all['site'].unique())
selected_site = st.sidebar.selectbox(
    "Select Site",
    options=sites_in_data,
    format_func=lambda x: x.capitalize()
)

# Display thresholds
st.sidebar.header("⚠️ Alert Thresholds")
for low, high, color, msg, hex_color in ALERT_THRESHOLDS:
    low_str = f"{low:.0f}" if low > 0 else "0"
    high_str = f"{high:.0f}" if high < 100 else "+"
    st.sidebar.markdown(
        f"<span style='display:inline-block; width:20px; height:20px; "
        f"background-color:{hex_color}; border-radius:50%; margin-right:10px;'></span>"
        f"<strong>{color}</strong>: {low_str}–{high_str}°C",
        unsafe_allow_html=True
    )

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

#st.markdown("<h1 class='main-header'>🕋 Hajj Heat Risk Monitoring Dashboard</h1>", 
#            unsafe_allow_html=True)
st.markdown("<h1 class='main-header'>Hajj Heat Risk Monitoring Dashboard</h1>", 
            unsafe_allow_html=True)

# Filter data for selected time
current_data = df_all[df_all['timestamp'] == selected_time]

# =============================================================================
# TOP ROW: KEY METRICS
# =============================================================================

st.subheader(f"📊 Current Conditions: {selected_time.strftime('%Y-%m-%d %H:%M')}")

# Get data for selected site
site_current = current_data[current_data['site'] == selected_site]

cols = st.columns(5)

if not site_current.empty:
    row = site_current.iloc[0]
    twb_eff = row.get('twb_eff_mean', row.get('risk_mean', np.nan))
    twb_base = row.get('TWB_C', np.nan)
    
    # Calculate contributions if available
    urban = row.get('dust_penalty_c', 0) + row.get('stagnant_penalty_c', 0)
    crowd = row.get('crowd_penalty_c', 0)
    
    alert_color, alert_msg = get_alert_info(twb_eff)
    
    with cols[0]:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Effective T_wb</div>"
            f"<div class='metric-value' style='color:{get_alert_color(twb_eff)};'>{twb_eff:.1f}°C</div>"
            f"<div class='metric-label'>{alert_color} ALERT</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with cols[1]:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Baseline T_wb</div>"
            f"<div class='metric-value'>{twb_base:.1f}°C</div>"
            f"<div class='metric-label'>Meteorological</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with cols[2]:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Urban + Wind Effects</div>"
            f"<div class='metric-value' style='color:#FF6B6B;'>+{urban:.2f}°C</div>"
            f"<div class='metric-label'>{(urban/(twb_eff-twb_base)*100) if twb_eff>twb_base else 0:.0f}% of amplification</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with cols[3]:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Crowd Contribution</div>"
            f"<div class='metric-value' style='color:#4ECDC4;'>+{crowd:.2f}°C</div>"
            f"<div class='metric-label'>{(crowd/(twb_eff-twb_base)*100) if twb_eff>twb_base else 0:.0f}% of amplification</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with cols[4]:
        hajj_day = row.get('hajj_day', 0)
        day_names = ["-2", "-1", "0 (Arrival)", "1 (Arafat)", "2 (Eid)", "3", "4", "5"]
        day_name = day_names[hajj_day + 2] if -2 <= hajj_day <= 5 else f"Day {hajj_day}"
        
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Hajj Day</div>"
            f"<div class='metric-value'>{hajj_day}</div>"
            f"<div class='metric-label'>{day_name}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# Alert banner
if not site_current.empty:
    st.markdown(
        f"<div class='alert-box {alert_color.lower()}'>"
        f"<strong>{alert_color} ALERT:</strong> {alert_msg} at {selected_site.capitalize()}"
        f"</div>",
        unsafe_allow_html=True
    )

# =============================================================================
# MAP VISUALIZATION (Based on selected view)
# =============================================================================

st.subheader("🗺️ Operational Risk Map")

col1, col2 = st.columns([2, 1])

with col1:
    #st.markdown("### 🧭 Operational Zones (Voronoi, Concave Boundary)")
    #st.markdown("### 🗺️ Operational Risk Map")

    cell_df = load_cell_risk_data(selected_year)
    if cell_df is None or cell_df.empty:
        st.warning("Cell-level data missing; cannot build operational zones.")
    else:
        # Build/load persistent zones (concave boundary + Voronoi)

        alpha_by_site = {
            "MINA": 160.0,
            "HARAM": 200.0,
            "MUZDALIFAH": 140.0,
            "ARAFAT": 110.0,
        }

        zones_meta_df, zone_stats_df = compute_zones_and_polygons_for_time(
            cell_df=cell_df,
            selected_time=selected_time,
            zone_counts=ZONE_COUNTS,
            alpha_by_site=alpha_by_site,
        )

        if zone_stats_df.empty or zones_meta_df.empty:
            st.warning("No zone polygons/stats available for this time.")
        else:
            m = build_folium_zones_map(zones_meta_df, zone_stats_df, selected_time)
            st_folium(m, width=900, height=550)

            # # Optional: show a zone table under the map
            # st.markdown("#### 🔎 Zone Table (selected time)")
            # st.dataframe(
            #     zone_stats_df.sort_values(["max", "mean"], ascending=False),
            #     use_container_width=True,
            # )

            # NOTE: if you change alpha/zone counts, delete saved zones for that year:
            # data/zones/cell_zones_<year>.csv and data/zones/zones_meta_<year>.csv

with col2:
    #st.markdown("### 🚨 Current Risk Summary")
        
    if not current_data.empty:
        # Sort by risk level (highest first)
        risk_order = {'BLACK': 5, 'RED': 4, 'ORANGE': 3, 'YELLOW': 2, 'GREEN': 1, 'UNKNOWN': 0}
        current_data_copy = current_data.copy()
        current_data_copy['risk_score'] = current_data_copy.apply(
            lambda row: risk_order.get(get_alert_info(
                row.get('twb_eff_mean', row.get('risk_mean', 0)))[0], 0), 
            axis=1
        )
        sorted_sites = current_data_copy.sort_values('risk_score', ascending=False)

        for i, (_, row) in enumerate(sorted_sites.iterrows()):
            site = row['site']
            temp = row.get('twb_eff_mean', row.get('risk_mean', 0))
            alert, msg = get_alert_info(temp)
            color = get_alert_color(temp)
            
            # Set margin-top: 0 for first iteration, 5px for others. Margin-bottom always 10px
            margin_top = "0" if i == 0 else "5px"
            margin_bottom = "5px"
            
            st.markdown(
                f"<div style='background-color:{color}20; border-left:5px solid {color}; "
                f"padding:10px; margin:{margin_top} 0 {margin_bottom} 0; border-radius:5px;'>"
                f"<h6 style='margin:0; color:{color};'>{site.capitalize()}</h6>"
                f"<p style='margin:0 0; font-size:1.3rem; font-weight:bold;'>{temp:.1f}°C</p>"
                f"<p style='margin:0;'><strong>{alert}</strong>: {msg}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Quick action summary
        st.markdown("### ⚡ Immediate Actions")
        
        black_sites = current_data_copy[current_data_copy.apply(
            lambda row: get_alert_info(row.get('twb_eff_mean', row.get('risk_mean', 0)))[0] == 'BLACK', axis=1
        )]
        
        red_sites = current_data_copy[current_data_copy.apply(
            lambda row: get_alert_info(row.get('twb_eff_mean', row.get('risk_mean', 0)))[0] == 'RED', axis=1
        )]
        
        if not black_sites.empty:
            st.error(f"🚨 **BLACK ALERT**: {', '.join(black_sites['site'].values)} - SUSPEND ALL OUTDOOR ACTIVITIES")
        
        if not red_sites.empty:
            st.warning(f"⚠️ **RED ALERT**: {', '.join(red_sites['site'].values)} - Protect vulnerable pilgrims")
        
        if black_sites.empty and red_sites.empty:
            st.success("✅ No extreme alerts currently")
    else:
        st.warning("No data for selected time")


# =============================================================================
# TIME SERIES PLOTS
# =============================================================================

st.subheader("📈 Temporal Risk Evolution")

# Filter data for selected site
site_df = df_all[df_all['site'] == selected_site].sort_values('timestamp')

if not site_df.empty:
    # Create time series plot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"{selected_site.capitalize()} - Risk Evolution", 
                       f"{selected_site.capitalize()} - Hourly Pattern (Selected Day)"),
        shared_xaxes=False,
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Add trace for effective temperature
    fig.add_trace(
        go.Scatter(
            x=site_df['timestamp'],
            y=site_df['twb_eff_mean'] if 'twb_eff_mean' in site_df.columns else site_df['risk_mean'],
            mode='lines',
            name='T_wb^eff',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Add trace for baseline temperature if available
    if 'TWB_C' in site_df.columns:
        fig.add_trace(
            go.Scatter(
                x=site_df['timestamp'],
                y=site_df['TWB_C'],
                mode='lines',
                name='T_wb (baseline)',
                line=dict(color='blue', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Add alert threshold lines
    y_max = site_df['twb_eff_mean' if 'twb_eff_mean' in site_df.columns else 'risk_mean'].max()
    for low, high, color, msg, hex_color in ALERT_THRESHOLDS:
        if high < 100 and low < y_max:
            fig.add_hline(
                y=high, 
                line_dash="dash", 
                line_color=hex_color,
                opacity=0.5,
                row=1, col=1
            )
    
    # Add hourly pattern for selected date
    date_mask = site_df['timestamp'].dt.date == selected_date
    day_df = site_df[date_mask].copy()
    
    if not day_df.empty:
        day_df = day_df.sort_values('hour')
        
        fig.add_trace(
            go.Scatter(
                x=day_df['hour'],
                y=day_df['twb_eff_mean'] if 'twb_eff_mean' in day_df.columns else day_df['risk_mean'],
                mode='lines+markers',
                name=f'{selected_date.strftime("%Y-%m-%d")}',
                line=dict(color='darkred', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        # Add baseline for comparison
        if 'TWB_C' in day_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=day_df['hour'],
                    y=day_df['TWB_C'],
                    mode='lines',
                    name='Baseline',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=2, col=1
            )
        
        # Add threshold lines to hourly plot
        for low, high, color, msg, hex_color in ALERT_THRESHOLDS:
            if high < 100:
                fig.add_hline(
                    y=high, 
                    line_dash="dash", 
                    line_color=hex_color,
                    opacity=0.3,
                    row=2, col=1
                )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=1, dtick=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# SITE COMPARISON
# =============================================================================

st.subheader("🔍 Site Comparison")

col1, col2 = st.columns(2)

with col1:
    # Current conditions across sites
    current_summary = current_data.copy()
    if not current_summary.empty:
        temp_col = 'twb_eff_mean' if 'twb_eff_mean' in current_summary.columns else 'risk_mean'
        current_summary['alert'] = current_summary[temp_col].apply(get_alert_level)
        
        # Prepare data for bar chart
        plot_df = current_summary[['site', 'TWB_C', temp_col]].copy()
        plot_df.columns = ['site', 'Baseline', 'Effective']
        plot_df = plot_df.melt(id_vars=['site'], var_name='Type', value_name='Temperature')
        
        fig = px.bar(
            plot_df,
            x='site',
            y='Temperature',
            color='Type',
            title="Current Temperatures by Site",
            barmode='group',
            color_discrete_map={
                'Baseline': 'blue',
                'Effective': 'red'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Alert distribution for selected site
    if not site_df.empty:
        site_df_copy = site_df.copy()
        site_df_copy['alert'] = site_df_copy['twb_eff_mean' if 'twb_eff_mean' in site_df_copy.columns else 'risk_mean'].apply(get_alert_level)
        alert_counts = site_df_copy['alert'].value_counts().reset_index()
        alert_counts.columns = ['Alert Level', 'Count']
        
        # Ensure all alert levels appear
        all_alerts = ["GREEN", "YELLOW", "ORANGE", "RED", "BLACK"]
        alert_colors = ["#28a745", "#ffc107", "#fd7e14", "#dc3545", "#000000"]
        
        fig = px.pie(
            alert_counts,
            values="Count",
            names="Alert Level",
            title=f"Alert Distribution - {selected_site.capitalize()} ({selected_year})",
            color="Alert Level",
            color_discrete_map={alert: color for alert, color in zip(all_alerts, alert_colors)}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# RAW DATA EXPORT
# =============================================================================

with st.expander("📥 Export Data"):
    st.markdown("Download current data for further analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not current_data.empty:
            csv = current_data.to_csv(index=False)
            st.download_button(
                label="📊 Download Current Conditions (CSV)",
                data=csv,
                file_name=f"hajj_risk_{selected_time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if not site_df.empty:
            site_csv = site_df.to_csv(index=False)
            st.download_button(
                label=f"📈 Download {selected_site} History (CSV)",
                data=site_csv,
                file_name=f"{selected_site.lower()}_{selected_year}_history.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #6c757d;'>"
    # f"🕋 Hajj Heat Risk Monitoring Dashboard | Based on physics-informed hybrid modeling | "
    f"Hajj Heat Risk Monitoring Dashboard | Based on physics-informed hybrid modeling | "
    f"Year: {selected_year} | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
    unsafe_allow_html=True
)