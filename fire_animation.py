#!/usr/bin/env python3
"""
Brunswick Creek V10742 + Ainslie Creek V10755
HTML-only NASA FIRMS animation.

Creates only:
    site/index.html

No PNG, GIF, or MP4 files are generated.
Required GitHub secret: FIRMS_MAP_KEY
"""

import io
import json
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TITLE = "Brunswick Complex Wildfire"
SUBTITLE = "Brunswick Creek (V10742) + Ainslie Creek (V10755)"
TZ = ZoneInfo("America/Vancouver")
START_DATE = date(2026, 7, 2)
BBOX = (-122.20, 49.40, -120.70, 50.70)  # west, south, east, north

SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
]

FRAME_MS = 750

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SITE_DIR = ROOT / "site"
CACHE_FILE = DATA_DIR / "firms_history.csv"

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

PLACES = {
    "Boston Bar": (-121.441, 49.864),
    "Hells Gate": (-121.420, 49.779),
    "Lytton": (-121.583, 50.231),
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Brunswick-HTML-Only/1.0"})
retry = Retry(
    total=5,
    connect=5,
    read=5,
    status=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["GET"]),
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)


def log(msg):
    print(msg, flush=True)


def today_local():
    return datetime.now(TZ).date()


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SITE_DIR.mkdir(parents=True, exist_ok=True)

    # Remove any old website media from previous versions.
    for pattern in ("*.png", "*.gif", "*.mp4"):
        for p in SITE_DIR.glob(pattern):
            p.unlink()

    old_frames = SITE_DIR / "frames"
    if old_frames.exists():
        shutil.rmtree(old_frames)


def get_key():
    key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not key:
        raise RuntimeError("FIRMS_MAP_KEY is missing.")
    return key


def read_cache():
    if not CACHE_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(CACHE_FILE)
        if not df.empty:
            df["acq_date"] = pd.to_datetime(
                df["acq_date"], errors="coerce"
            ).dt.date
        return df
    except Exception as exc:
        log(f"Cache warning: {exc}")
        return pd.DataFrame()


def fetch_chunk(key, source, start, days):
    west, south, east, north = BBOX
    area = f"{west},{south},{east},{north}"
    url = f"{FIRMS_BASE}/{key}/{source}/{area}/{days}/{start.isoformat()}"

    last_error = None

    for attempt in range(1, 5):
        try:
            log(f"Downloading {source}: {start} for {days} day(s)")
            r = SESSION.get(url, timeout=120)
            r.raise_for_status()
            text = r.text.strip()

            if not text:
                return pd.DataFrame()

            if "latitude" not in text.lower() or "longitude" not in text.lower():
                raise RuntimeError(text[:250].replace("\n", " "))

            df = pd.read_csv(io.StringIO(text))
            if df.empty:
                return df

            df.columns = [str(c).strip() for c in df.columns]
            df["source"] = source
            df["acq_date"] = pd.to_datetime(
                df["acq_date"], errors="coerce"
            ).dt.date
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df = df.dropna(subset=["latitude", "longitude", "acq_date"])

            return df[
                df["longitude"].between(west, east)
                & df["latitude"].between(south, north)
            ].copy()

        except Exception as exc:
            last_error = exc
            log(f"{source} attempt {attempt}/4 failed: {exc}")
            if attempt < 4:
                time.sleep(attempt * 5)

    raise RuntimeError(f"{source} failed: {last_error}")


def fetch_recent(key, cached, today):
    if cached.empty or cached["acq_date"].dropna().empty:
        start = START_DATE
    else:
        latest = max(cached["acq_date"].dropna())
        start = max(START_DATE, latest - timedelta(days=2))

    if start > today:
        return pd.DataFrame()

    chunks = []
    current = start

    while current <= today:
        days = min(5, (today - current).days + 1)

        for source in SOURCES:
            try:
                part = fetch_chunk(key, source, current, days)
                if not part.empty:
                    chunks.append(part)
            except Exception as exc:
                log(f"Warning: {source} failed for {current}: {exc}")

        current += timedelta(days=days)

    if not chunks:
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True, sort=False)


def normalize(cached, new_data, today):
    frames = [x for x in (cached, new_data) if not x.empty]

    if not frames:
        raise RuntimeError("No FIRMS data available.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce").dt.date
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "acq_date"])

    dedup = [
        c for c in ("source", "latitude", "longitude", "acq_date", "acq_time")
        if c in df.columns
    ]
    if dedup:
        df = df.drop_duplicates(subset=dedup, keep="last")

    df = df[
        (df["acq_date"] >= START_DATE)
        & (df["acq_date"] <= today)
    ].sort_values(
        ["acq_date", "latitude", "longitude"]
    ).reset_index(drop=True)

    save = df.copy()
    save["acq_date"] = save["acq_date"].astype(str)
    save.to_csv(CACHE_FILE, index=False)

    return df


def build_html(df, today):
    dates = []
    d = START_DATE
    while d <= today:
        dates.append(d.isoformat())
        d += timedelta(days=1)

    date_index = {
        date.fromisoformat(value): i
        for i, value in enumerate(dates)
    }

    detections = []
    for row in df[["latitude", "longitude", "acq_date"]].itertuples(index=False):
        idx = date_index.get(row.acq_date)
        if idx is not None:
            detections.append([
                round(float(row.latitude), 5),
                round(float(row.longitude), 5),
                int(idx),
            ])

    places = [
        {"name": name, "lon": lon, "lat": lat}
        for name, (lon, lat) in PLACES.items()
    ]

    dates_json = json.dumps(dates, separators=(",", ":"))
    detections_json = json.dumps(detections, separators=(",", ":"))
    places_json = json.dumps(places, separators=(",", ":"))

    west, south, east, north = BBOX

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#000000">
<title>{TITLE}</title>
<link rel="stylesheet"
 href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
 crossorigin="">
<style>
html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#000;font-family:Arial,sans-serif}}
#map{{position:absolute;inset:0;background:#000}}
#fireCanvas{{position:absolute;inset:0;z-index:450;pointer-events:none}}
#heading{{position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:900;
max-width:calc(100% - 24px);background:rgba(255,255,255,.9);border-radius:10px;padding:8px 12px;text-align:center}}
#heading strong{{display:block;font-size:clamp(16px,2.4vw,23px)}}
#dateLabel{{margin-top:3px;color:darkblue;font-size:clamp(18px,3vw,28px);font-weight:bold}}
#controls{{position:fixed;right:12px;bottom:12px;z-index:1000;display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}}
button,select{{min-height:44px;border:0;border-radius:10px;font-size:16px;padding:10px 14px}}
button{{background:rgba(0,0,0,.68);color:white}}
select{{background:white;color:black}}
#counter{{position:fixed;left:12px;bottom:12px;z-index:1000;color:white;background:rgba(0,0,0,.68);
border-radius:10px;padding:10px 12px;font-size:14px}}
#note{{position:fixed;left:50%;bottom:12px;transform:translateX(-50%);z-index:900;max-width:52%;
background:rgba(255,255,255,.82);border-radius:8px;padding:6px 9px;text-align:center;font-size:11px}}
@media(max-width:760px){{
 #controls{{left:8px;right:8px;bottom:8px;justify-content:center}}
 #counter{{left:8px;bottom:66px}}
 #note{{display:none}}
}}
</style>
</head>
<body>
<div id="map"></div>
<canvas id="fireCanvas"></canvas>

<div id="heading">
  <strong>{TITLE}</strong>
  <span>{SUBTITLE}</span>
  <div id="dateLabel"></div>
</div>

<div id="counter"></div>
<div id="note">Red = cumulative NASA FIRMS thermal detections. Hotspots are not official wildfire perimeters.</div>

<div id="controls">
  <button id="previous" type="button">Previous</button>
  <button id="playPause" type="button">Pause</button>
  <button id="next" type="button">Next</button>
  <select id="speed" aria-label="Animation speed">
    <option value="{FRAME_MS * 2}">Slow</option>
    <option value="{FRAME_MS}" selected>Normal</option>
    <option value="{max(250, FRAME_MS // 2)}">Fast</option>
  </select>
  <button id="fs" type="button">Full screen</button>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
<script>
const frameDates={dates_json};
const detections={detections_json};
const places={places_json};

const map=L.map("map",{{zoomControl:true,preferCanvas:true}});
map.fitBounds([[{south},{west}],[{north},{east}]]);

L.tileLayer(
 "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}",
 {{maxZoom:18,attribution:"Esri World Topographic Map"}}
).addTo(map);

for(const p of places){{
 L.circleMarker([p.lat,p.lon],{{radius:3,color:"#000",fillColor:"#000",fillOpacity:1,weight:1}})
  .bindTooltip(p.name,{{permanent:true,direction:"right"}}).addTo(map);
}}

const canvas=document.getElementById("fireCanvas");
const ctx=canvas.getContext("2d");
const dateLabel=document.getElementById("dateLabel");
const counter=document.getElementById("counter");
const previous=document.getElementById("previous");
const playPause=document.getElementById("playPause");
const next=document.getElementById("next");
const speed=document.getElementById("speed");
const fs=document.getElementById("fs");

let frameIndex=0;
let playing=true;
let timer=null;

function formatDate(value){{
 const p=value.split("-");
 return new Date(Number(p[0]),Number(p[1])-1,Number(p[2])).toLocaleDateString(
  "en-CA",{{year:"numeric",month:"long",day:"numeric"}}
 );
}}

function resizeCanvas(){{
 const rect=document.getElementById("map").getBoundingClientRect();
 const ratio=window.devicePixelRatio||1;
 canvas.style.width=rect.width+"px";
 canvas.style.height=rect.height+"px";
 canvas.width=Math.round(rect.width*ratio);
 canvas.height=Math.round(rect.height*ratio);
 ctx.setTransform(ratio,0,0,ratio,0,0);
 redraw();
}}

function redraw(){{
 const rect=canvas.getBoundingClientRect();
 ctx.clearRect(0,0,rect.width,rect.height);
 ctx.fillStyle="rgba(255,0,0,.72)";

 for(const d of detections){{
  if(d[2]>frameIndex) continue;
  const pt=map.latLngToContainerPoint([d[0],d[1]]);
  if(pt.x<-5||pt.y<-5||pt.x>rect.width+5||pt.y>rect.height+5) continue;
  ctx.beginPath();
  ctx.arc(pt.x,pt.y,2.8,0,Math.PI*2);
  ctx.fill();
 }}

 dateLabel.textContent=formatDate(frameDates[frameIndex]);
 counter.textContent="Frame "+(frameIndex+1)+" of "+frameDates.length;
}}

function stopTimer(){{
 if(timer!==null){{clearTimeout(timer);timer=null}}
}}

function scheduleNext(){{
 stopTimer();
 if(!playing) return;
 if(frameIndex>=frameDates.length-1){{
  playing=false;
  playPause.textContent="Play";
  return;
 }}
 timer=setTimeout(()=>{{
  frameIndex+=1;
  redraw();
  scheduleNext();
 }},Number(speed.value));
}}

function manualPause(){{
 playing=false;
 playPause.textContent="Play";
 stopTimer();
}}

previous.addEventListener("click",()=>{{
 manualPause();
 frameIndex=Math.max(0,frameIndex-1);
 redraw();
}});

next.addEventListener("click",()=>{{
 manualPause();
 frameIndex=Math.min(frameDates.length-1,frameIndex+1);
 redraw();
}});

playPause.addEventListener("click",()=>{{
 if(playing){{manualPause();return}}
 if(frameIndex>=frameDates.length-1){{frameIndex=0;redraw()}}
 playing=true;
 playPause.textContent="Pause";
 scheduleNext();
}});

speed.addEventListener("change",()=>{{if(playing)scheduleNext()}});

fs.addEventListener("click",async()=>{{
 try{{
  if(!document.fullscreenElement) await document.documentElement.requestFullscreen();
  else await document.exitFullscreen();
 }}catch(e){{}}
}});

map.on("move zoom resize",redraw);
window.addEventListener("resize",resizeCanvas);

resizeCanvas();
redraw();
scheduleNext();
</script>
</body>
</html>
"""

    output = SITE_DIR / "index.html"
    output.write_text(html, encoding="utf-8")

    log(f"Created HTML-only animation: {output}")
    log(f"Embedded {len(detections):,} detections across {len(dates)} frames.")


def main():
    ensure_dirs()
    key = get_key()
    today = today_local()

    if today < START_DATE:
        raise RuntimeError("Current date is before the animation start date.")

    cached = read_cache()
    new_data = fetch_recent(key, cached, today)
    df = normalize(cached, new_data, today)

    log(f"Total FIRMS detections: {len(df):,}")
    build_html(df, today)
    log("DONE — Brunswick HTML-only animation updated successfully.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
