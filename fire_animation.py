#!/usr/bin/env python3
"""
Brunswick Complex wildfire animation
Brunswick Creek (V10742) + Ainslie Creek (V10755)

Creates cumulative NASA FIRMS hotspot frames from July 2, 2026 through today,
plus GIF/MP4 files, a latest PNG, and a GitHub Pages-ready slideshow.

Web controls match the BC History viewer:
    Previous
    Play / Pause
    Next
    Slow / Normal / Fast
    Full screen

The animation auto-plays once and then holds on the final frame.

Required GitHub secret/environment variable:
    FIRMS_MAP_KEY
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image

# -----------------------------------------------------------------------------
# BRUNSWICK SETTINGS
# -----------------------------------------------------------------------------

FIRE_TITLE = "Brunswick Complex Wildfire"
FIRE_SUBTITLE = "Brunswick Creek (V10742) + Ainslie Creek (V10755)"

START_DATE = date(2026, 7, 2)

# Wide enough to show Brunswick Creek west of the Fraser and
# Ainslie Creek east.
# FIRMS order: west, south, east, north.
BBOX = (-122.20, 49.40, -120.70, 50.70)

SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
]

CUMULATIVE = True

FIRE_COLOR = "red"
FIRE_POINT_SIZE = 22
FIRE_POINT_ALPHA = 0.78

DATE_FONT_SIZE = 26
DATE_COLOR = "darkblue"

FRAME_DPI = 150
FRAME_SECONDS = 0.75
LAST_FRAME_SECONDS = 8.0

EXPANSION_TOLERANCE_METERS = 500
NO_EXPANSION_DAYS_TO_STOP = 5

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
FRAMES_DIR = ROOT / "frames"
SITE_DIR = ROOT / "site"
SITE_FRAMES_DIR = SITE_DIR / "frames"

CACHE_FILE = DATA_DIR / "firms_history.csv"
STATE_FILE = DATA_DIR / "fire_status.json"
STOP_FLAG = ROOT / "stop_updates.flag"

GIF_FILE = ROOT / "brunswick_fire_animation.gif"
MP4_FILE = ROOT / "brunswick_fire_animation.mp4"
LATEST_FILE = ROOT / "latest.png"

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

SESSION = requests.Session()
SESSION.headers.update(
    {"User-Agent": "Brunswick-Wildfire-Animation/1.0"}
)


# -----------------------------------------------------------------------------
# BASIC HELPERS
# -----------------------------------------------------------------------------

def log(message: str) -> None:
    print(message, flush=True)


def daterange(start: date, end: date) -> Iterable[date]:
    current = start

    while current <= end:
        yield current
        current += timedelta(days=1)


def ensure_dirs() -> None:
    for folder in (
        DATA_DIR,
        FRAMES_DIR,
        SITE_DIR,
        SITE_FRAMES_DIR,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


def get_map_key() -> str:
    key = os.getenv(
        "FIRMS_MAP_KEY",
        "",
    ).strip()

    if not key:
        raise RuntimeError(
            "FIRMS_MAP_KEY is missing. "
            "In GitHub: Settings > Secrets and variables > "
            "Actions > Repository secrets, add FIRMS_MAP_KEY."
        )

    return key


# -----------------------------------------------------------------------------
# FIRMS DATA
# -----------------------------------------------------------------------------

def read_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            CACHE_FILE
        )

        if (
            not df.empty
            and "acq_date" in df.columns
        ):
            df["acq_date"] = pd.to_datetime(
                df["acq_date"],
                errors="coerce",
            ).dt.date

        return df

    except Exception as exc:
        log(
            f"Warning: could not read cache: {exc}"
        )
        return pd.DataFrame()


def firms_url(
    map_key: str,
    source: str,
    start: date,
    days: int,
) -> str:
    west, south, east, north = BBOX
    area = (
        f"{west},{south},{east},{north}"
    )

    return (
        f"{FIRMS_BASE}/"
        f"{map_key}/"
        f"{source}/"
        f"{area}/"
        f"{days}/"
        f"{start.isoformat()}"
    )


def fetch_chunk(
    map_key: str,
    source: str,
    start: date,
    days: int,
) -> pd.DataFrame:
    url = firms_url(
        map_key,
        source,
        start,
        days,
    )

    log(
        f"Downloading {source}: "
        f"{start.isoformat()} for {days} day(s)"
    )

    response = SESSION.get(
        url,
        timeout=90,
    )
    response.raise_for_status()

    text = response.text.strip()

    if not text:
        return pd.DataFrame()

    if (
        "latitude" not in text.lower()
        or "longitude" not in text.lower()
    ):
        preview = text[:300].replace(
            "\n",
            " ",
        )

        raise RuntimeError(
            f"FIRMS returned an unexpected response "
            f"for {source}: {preview}"
        )

    df = pd.read_csv(
        io.StringIO(text)
    )

    if df.empty:
        return df

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    required = {
        "latitude",
        "longitude",
        "acq_date",
    }

    if not required.issubset(
        df.columns
    ):
        raise RuntimeError(
            f"FIRMS response is missing columns: "
            f"{required - set(df.columns)}"
        )

    df["source"] = source

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    ).dt.date

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date",
        ]
    )

    west, south, east, north = BBOX

    df = df[
        df["longitude"].between(
            west,
            east,
        )
        &
        df["latitude"].between(
            south,
            north,
        )
    ].copy()

    return df


def fetch_missing_data(
    map_key: str,
    cached: pd.DataFrame,
    today: date,
) -> pd.DataFrame:
    """
    Fetch only dates not safely covered by the cache,
    with a one-day overlap because NRT data can be revised.
    """

    if (
        cached.empty
        or "acq_date" not in cached.columns
        or cached["acq_date"].dropna().empty
    ):
        fetch_start = START_DATE

    else:
        latest_cached = max(
            cached["acq_date"].dropna()
        )

        fetch_start = max(
            START_DATE,
            latest_cached - timedelta(days=1),
        )

    if fetch_start > today:
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []

    current = fetch_start

    while current <= today:
        days = min(
            5,
            (today - current).days + 1,
        )

        for source in SOURCES:
            try:
                part = fetch_chunk(
                    map_key,
                    source,
                    current,
                    days,
                )

                if not part.empty:
                    chunks.append(
                        part
                    )

            except Exception as exc:
                log(
                    f"Warning: {source} failed "
                    f"for {current}: {exc}"
                )

        current += timedelta(
            days=days
        )

    if not chunks:
        return pd.DataFrame()

    return pd.concat(
        chunks,
        ignore_index=True,
        sort=False,
    )


def normalize_and_save(
    cached: pd.DataFrame,
    new_data: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        frame
        for frame in (
            cached,
            new_data,
        )
        if not frame.empty
    ]

    if not frames:
        return pd.DataFrame(
            columns=[
                "latitude",
                "longitude",
                "acq_date",
                "source",
            ]
        )

    df = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    df["acq_date"] = pd.to_datetime(
        df["acq_date"],
        errors="coerce",
    ).dt.date

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "acq_date",
        ]
    )

    dedup_cols = [
        column
        for column in [
            "source",
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
        ]
        if column in df.columns
    ]

    if dedup_cols:
        df = df.drop_duplicates(
            subset=dedup_cols,
            keep="last",
        )

    df = df[
        df["acq_date"] >= START_DATE
    ].copy()

    df = df.sort_values(
        [
            "acq_date",
            "latitude",
            "longitude",
        ]
    ).reset_index(
        drop=True
    )

    save_df = df.copy()

    save_df["acq_date"] = (
        save_df["acq_date"].astype(str)
    )

    save_df.to_csv(
        CACHE_FILE,
        index=False,
    )

    return df


# -----------------------------------------------------------------------------
# MAP
# -----------------------------------------------------------------------------

def add_basemap(ax) -> None:
    try:
        import contextily as ctx

        ctx.add_basemap(
            ax,
            source=ctx.providers.Esri.WorldTopoMap,
            crs="EPSG:4326",
            attribution=False,
            zoom=8,
        )

    except Exception as exc:
        log(
            f"Basemap warning "
            f"(continuing without tiles): {exc}"
        )

        ax.set_facecolor(
            "#e8efe3"
        )


def add_reference_labels(ax) -> None:
    places = {
        "Boston Bar": (
            -121.441,
            49.864,
        ),
        "Hells Gate": (
            -121.420,
            49.779,
        ),
        "Lytton": (
            -121.583,
            50.231,
        ),
    }

    west, south, east, north = BBOX

    for name, (
        lon,
        lat,
    ) in places.items():
        if (
            west <= lon <= east
            and south <= lat <= north
        ):
            ax.plot(
                lon,
                lat,
                marker="o",
                markersize=3.5,
                color="black",
                zorder=6,
            )

            ax.text(
                lon + 0.018,
                lat + 0.010,
                name,
                fontsize=8,
                color="black",
                fontweight="bold",
                zorder=7,
                bbox=dict(
                    facecolor="white",
                    alpha=0.65,
                    edgecolor="none",
                    pad=1.5,
                ),
            )


def frame_data_for_date(
    df: pd.DataFrame,
    frame_date: date,
) -> pd.DataFrame:
    if df.empty:
        return df

    if CUMULATIVE:
        return df[
            df["acq_date"]
            <=
            frame_date
        ]

    return df[
        df["acq_date"]
        ==
        frame_date
    ]


def make_frame(
    df: pd.DataFrame,
    frame_date: date,
    output_path: Path,
) -> None:
    west, south, east, north = BBOX

    day_df = frame_data_for_date(
        df,
        frame_date,
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    ax.set_xlim(
        west,
        east,
    )

    ax.set_ylim(
        south,
        north,
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    add_basemap(
        ax
    )

    if not day_df.empty:
        ax.scatter(
            day_df["longitude"],
            day_df["latitude"],
            s=FIRE_POINT_SIZE,
            c=FIRE_COLOR,
            alpha=FIRE_POINT_ALPHA,
            linewidths=0,
            zorder=8,
        )

    add_reference_labels(
        ax
    )

    ax.text(
        0.5,
        0.975,
        frame_date.strftime(
            "%B %d, %Y"
        ),
        transform=ax.transAxes,
        fontsize=DATE_FONT_SIZE,
        fontweight="bold",
        color=DATE_COLOR,
        ha="center",
        va="top",
        zorder=20,
        bbox=dict(
            facecolor="white",
            alpha=0.86,
            edgecolor="none",
            boxstyle="round,pad=0.35",
        ),
    )

    ax.text(
        0.5,
        1.015,
        FIRE_TITLE,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="bottom",
        color="black",
    )

    ax.text(
        0.5,
        -0.045,
        FIRE_SUBTITLE
        + "   •   Red = cumulative NASA FIRMS detections",
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        va="top",
        color="black",
    )

    ax.set_xlabel(
        "Longitude"
    )

    ax.set_ylabel(
        "Latitude"
    )

    ax.grid(
        True,
        alpha=0.16,
        linewidth=0.6,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.91,
        bottom=0.08,
    )

    fig.savefig(
        output_path,
        dpi=FRAME_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def build_frames(
    df: pd.DataFrame,
    today: date,
) -> list[Path]:
    for old in FRAMES_DIR.glob(
        "frame_*.png"
    ):
        old.unlink()

    frame_paths: list[Path] = []

    total = (
        today - START_DATE
    ).days + 1

    for index, frame_date in enumerate(
        daterange(
            START_DATE,
            today,
        ),
        start=1,
    ):
        path = (
            FRAMES_DIR
            /
            f"frame_{frame_date.isoformat()}.png"
        )

        log(
            f"Creating frame "
            f"{index} of {total}: "
            f"{frame_date}"
        )

        make_frame(
            df,
            frame_date,
            path,
        )

        frame_paths.append(
            path
        )

    if not frame_paths:
        raise RuntimeError(
            "No frames were created."
        )

    shutil.copy2(
        frame_paths[-1],
        LATEST_FILE,
    )

    shutil.copy2(
        frame_paths[-1],
        ROOT / "brunswick_latest.png",
    )

    return frame_paths


# -----------------------------------------------------------------------------
# GIF + MP4
# -----------------------------------------------------------------------------

def build_gif(
    frame_paths: list[Path],
) -> None:
    images = [
        imageio.imread(path)
        for path in frame_paths
    ]

    durations = [
        FRAME_SECONDS
    ] * len(images)

    durations[-1] = (
        LAST_FRAME_SECONDS
    )

    imageio.mimsave(
        GIF_FILE,
        images,
        duration=durations,
        loop=0,
    )

    shutil.copy2(
        GIF_FILE,
        ROOT / "wildfire_animation.gif",
    )

    shutil.copy2(
        GIF_FILE,
        ROOT / "animation.gif",
    )

    log(
        f"Created {GIF_FILE.name}"
    )


def build_mp4(
    frame_paths: list[Path],
) -> None:
    try:
        fps = max(
            1,
            round(
                1 / FRAME_SECONDS
            ),
        )

        with imageio.get_writer(
            MP4_FILE,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        ) as writer:

            for path in frame_paths:
                frame = imageio.imread(
                    path
                )

                h, w = frame.shape[:2]

                if h % 2 or w % 2:
                    img = Image.fromarray(
                        frame
                    )

                    img = img.crop(
                        (
                            0,
                            0,
                            w - (w % 2),
                            h - (h % 2),
                        )
                    )

                    frame = np.asarray(
                        img
                    )

                writer.append_data(
                    frame
                )

            last = imageio.imread(
                frame_paths[-1]
            )

            h, w = last.shape[:2]

            if h % 2 or w % 2:
                img = Image.fromarray(
                    last
                )

                img = img.crop(
                    (
                        0,
                        0,
                        w - (w % 2),
                        h - (h % 2),
                    )
                )

                last = np.asarray(
                    img
                )

            extra = int(
                LAST_FRAME_SECONDS
                * fps
            )

            for _ in range(extra):
                writer.append_data(
                    last
                )

        shutil.copy2(
            MP4_FILE,
            ROOT / "wildfire_animation.mp4",
        )

        shutil.copy2(
            MP4_FILE,
            ROOT / "animation.mp4",
        )

        log(
            f"Created {MP4_FILE.name}"
        )

    except Exception as exc:
        log(
            f"MP4 warning: {exc}"
        )


# -----------------------------------------------------------------------------
# GITHUB PAGES VIEWER
# -----------------------------------------------------------------------------

def build_website(
    frame_paths: list[Path],
) -> None:
    """
    Same controls as BC History:
      Previous
      Play / Pause
      Next
      Slow / Normal / Fast
      Full screen

    Auto-plays once and holds on the last frame.
    """

    log(
        "Building Brunswick GitHub Pages viewer..."
    )

    if SITE_FRAMES_DIR.exists():
        shutil.rmtree(
            SITE_FRAMES_DIR
        )

    SITE_FRAMES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    web_names: list[str] = []

    for path in frame_paths:
        dest = (
            SITE_FRAMES_DIR
            /
            path.name
        )

        shutil.copy2(
            path,
            dest,
        )

        web_names.append(
            f"frames/{path.name}"
        )

    if GIF_FILE.exists():
        shutil.copy2(
            GIF_FILE,
            SITE_DIR / GIF_FILE.name,
        )

    if MP4_FILE.exists():
        shutil.copy2(
            MP4_FILE,
            SITE_DIR / MP4_FILE.name,
        )

    shutil.copy2(
        LATEST_FILE,
        SITE_DIR / LATEST_FILE.name,
    )

    manifest = {
        "name": FIRE_TITLE,
        "short_name": "Brunswick Fire",
        "start_url": "./",
        "display": "standalone",
        "background_color": "#000000",
        "theme_color": "#000000",
    }

    (
        SITE_DIR
        / "manifest.webmanifest"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    frames_json = json.dumps(
        web_names
    )

    normal_delay = int(
        FRAME_SECONDS
        * 1000
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#000000">
<link rel="manifest" href="manifest.webmanifest">

<title>{FIRE_TITLE}</title>

<style>
html,
body {{
    margin: 0;
    width: 100%;
    height: 100%;
    background: #000;
    overflow: hidden;
}}

body {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: Arial, Helvetica, sans-serif;
}}

#frame {{
    display: block;
    width: 100vw;
    height: 100vh;
    max-width: none;
    max-height: none;
    object-fit: contain;
    object-position: center center;
    background: #000;
}}

#controls {{
    position: fixed;
    right: 12px;
    bottom: 12px;
    z-index: 30;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}}

button,
select {{
    min-height: 44px;
    border: 0;
    border-radius: 10px;
    font-size: 16px;
    padding: 10px 14px;
}}

button {{
    background: rgba(0,0,0,0.62);
    color: white;
}}

select {{
    background: white;
    color: black;
}}

#counter {{
    position: fixed;
    left: 12px;
    bottom: 12px;
    z-index: 25;
    color: white;
    background: rgba(0,0,0,0.62);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 14px;
}}

@media (max-width: 700px) {{
    #controls {{
        left: 8px;
        right: 8px;
        bottom: 8px;
        justify-content: center;
    }}

    #counter {{
        left: 8px;
        bottom: 66px;
    }}
}}
</style>
</head>

<body>

<img
    id="frame"
    alt="Brunswick wildfire animation"
>

<div
    id="counter"
    aria-live="polite"
></div>

<div id="controls">

    <button
        id="previous"
        type="button"
    >
        Previous
    </button>

    <button
        id="playPause"
        type="button"
    >
        Pause
    </button>

    <button
        id="next"
        type="button"
    >
        Next
    </button>

    <select
        id="speed"
        aria-label="Animation speed"
    >
        <option value="{normal_delay * 2}">
            Slow
        </option>

        <option
            value="{normal_delay}"
            selected
        >
            Normal
        </option>

        <option value="{max(250, normal_delay // 2)}">
            Fast
        </option>
    </select>

    <button
        id="fs"
        type="button"
    >
        Full screen
    </button>

</div>

<script>
const frames = {frames_json};

const img =
    document.getElementById("frame");

const counter =
    document.getElementById("counter");

const previous =
    document.getElementById("previous");

const playPause =
    document.getElementById("playPause");

const next =
    document.getElementById("next");

const speed =
    document.getElementById("speed");

const fs =
    document.getElementById("fs");

let index = 0;
let playing = true;
let timer = null;


function stopTimer() {{

    if (timer !== null) {{

        window.clearTimeout(
            timer
        );

        timer = null;
    }}
}}


function updateCounter() {{

    counter.textContent =
        "Frame "
        + (index + 1)
        + " of "
        + frames.length;
}}


function showFrame() {{

    if (!frames.length) {{
        return;
    }}

    img.src =
        frames[index];

    updateCounter();
}}


function scheduleNext() {{

    stopTimer();

    if (
        !playing
        ||
        !frames.length
    ) {{
        return;
    }}

    // Stop on the last frame.
    if (
        index
        >=
        frames.length - 1
    ) {{

        playing = false;

        playPause.textContent =
            "Play";

        return;
    }}

    timer =
        window.setTimeout(
            () => {{

                index += 1;

                showFrame();

                scheduleNext();

            }},
            Number(
                speed.value
            )
        );
}}


function pauseForManualMove() {{

    playing = false;

    playPause.textContent =
        "Play";

    stopTimer();
}}


previous.addEventListener(
    "click",
    () => {{

        pauseForManualMove();

        index =
            Math.max(
                0,
                index - 1
            );

        showFrame();
    }}
);


next.addEventListener(
    "click",
    () => {{

        pauseForManualMove();

        index =
            Math.min(
                frames.length - 1,
                index + 1
            );

        showFrame();
    }}
);


playPause.addEventListener(
    "click",
    () => {{

        if (!frames.length) {{
            return;
        }}

        if (playing) {{

            playing = false;

            playPause.textContent =
                "Play";

            stopTimer();

            return;
        }}

        // Pressing Play on the final frame
        // restarts from the beginning.
        if (
            index
            >=
            frames.length - 1
        ) {{

            index = 0;

            showFrame();
        }}

        playing = true;

        playPause.textContent =
            "Pause";

        scheduleNext();
    }}
);


speed.addEventListener(
    "change",
    () => {{

        if (playing) {{
            scheduleNext();
        }}
    }}
);


fs.addEventListener(
    "click",
    async () => {{

        try {{

            if (
                !document.fullscreenElement
            ) {{

                await document
                    .documentElement
                    .requestFullscreen();

            }} else {{

                await document
                    .exitFullscreen();
            }}

        }} catch (e) {{

        }}
    }}
);


showFrame();

scheduleNext();

</script>

</body>
</html>
"""

    (
        SITE_DIR
        / "index.html"
    ).write_text(
        html,
        encoding="utf-8",
    )

    # Root copy for simple Pages setups.
    root_html = html.replace(
        "frames/",
        "site/frames/",
    )

    (
        ROOT
        / "index.html"
    ).write_text(
        root_html,
        encoding="utf-8",
    )

    log(
        "Created Brunswick web viewer with "
        "Previous / Play-Pause / Next / "
        "Speed / Full-screen controls."
    )


# -----------------------------------------------------------------------------
# AUTO-STOP LOGIC
# -----------------------------------------------------------------------------

def haversine_meters(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius = 6_371_000.0

    p1 = math.radians(
        lat1
    )

    p2 = math.radians(
        lat2
    )

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(
            dphi / 2
        ) ** 2
        +
        math.cos(p1)
        *
        math.cos(p2)
        *
        math.sin(
            dlambda / 2
        ) ** 2
    )

    return (
        2
        *
        radius
        *
        math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a),
        )
    )


def day_has_expansion(
    df: pd.DataFrame,
    check_date: date,
) -> bool:
    if df.empty:
        return False

    todays = df[
        df["acq_date"]
        ==
        check_date
    ]

    prior = df[
        df["acq_date"]
        <
        check_date
    ]

    if todays.empty:
        return False

    if prior.empty:
        return True

    prior_points = list(
        prior.assign(
            lat_key=prior[
                "latitude"
            ].round(3),

            lon_key=prior[
                "longitude"
            ].round(3),
        )[
            [
                "lat_key",
                "lon_key",
            ]
        ]
        .drop_duplicates()
        .itertuples(
            index=False,
            name=None,
        )
    )

    for row in todays[
        [
            "latitude",
            "longitude",
        ]
    ].itertuples(
        index=False
    ):
        expanded = True

        for plat, plon in prior_points:

            if (
                haversine_meters(
                    row.latitude,
                    row.longitude,
                    plat,
                    plon,
                )
                <=
                EXPANSION_TOLERANCE_METERS
            ):
                expanded = False
                break

        if expanded:
            return True

    return False


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {
            "last_checked_date": None,
            "no_expansion_days": 0,
        }

    try:
        return json.loads(
            STATE_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return {
            "last_checked_date": None,
            "no_expansion_days": 0,
        }


def update_stop_state(
    df: pd.DataFrame,
    today: date,
) -> None:
    state = load_state()

    today_s = today.isoformat()

    if (
        state.get(
            "last_checked_date"
        )
        ==
        today_s
    ):
        log(
            f"Expansion state already checked "
            f"for {today_s}: "
            f"{state.get('no_expansion_days', 0)} "
            f"no-expansion day(s)."
        )

        return

    expanded = day_has_expansion(
        df,
        today,
    )

    if expanded:
        state[
            "no_expansion_days"
        ] = 0

        if STOP_FLAG.exists():
            STOP_FLAG.unlink()

        log(
            "New fire expansion detected today."
        )

    else:
        state[
            "no_expansion_days"
        ] = (
            int(
                state.get(
                    "no_expansion_days",
                    0,
                )
            )
            + 1
        )

        log(
            f"No new expansion today. "
            f"Consecutive days: "
            f"{state['no_expansion_days']}"
        )

    state[
        "last_checked_date"
    ] = today_s

    state[
        "checked_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    state[
        "expansion_tolerance_m"
    ] = (
        EXPANSION_TOLERANCE_METERS
    )

    STATE_FILE.write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )

    if (
        state[
            "no_expansion_days"
        ]
        >=
        NO_EXPANSION_DAYS_TO_STOP
    ):
        STOP_FLAG.write_text(
            (
                "No fire expansion detected for "
                f"{state['no_expansion_days']} "
                "consecutive days.\n"
            ),
            encoding="utf-8",
        )

        log(
            "Created stop_updates.flag — "
            "automatic updates may now be stopped "
            "by the workflow."
        )


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main() -> int:
    ensure_dirs()

    map_key = get_map_key()

    today = date.today()

    if today < START_DATE:
        raise RuntimeError(
            f"Current date {today} is before "
            f"animation start date {START_DATE}."
        )

    cached = read_cache()

    new_data = fetch_missing_data(
        map_key,
        cached,
        today,
    )

    df = normalize_and_save(
        cached,
        new_data,
    )

    log(
        f"Total cached FIRMS detections: "
        f"{len(df):,}"
    )

    if not df.empty:
        log(
            f"Detection dates: "
            f"{df['acq_date'].min()} "
            f"through "
            f"{df['acq_date'].max()}"
        )

    frame_paths = build_frames(
        df,
        today,
    )

    build_gif(
        frame_paths
    )

    build_mp4(
        frame_paths
    )

    build_website(
        frame_paths
    )

    update_stop_state(
        df,
        today,
    )

    log(
        "DONE — Brunswick wildfire animation "
        "updated successfully."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
            flush=True,
        )

        raise
