#!/usr/bin/env python3

from __future__ import annotations

import base64
import io
import json
import os
import shutil

from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import contextily as ctx
import geopandas as gpd
import imageio.v2 as imageio

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from PIL import Image
from shapely.geometry import Point
from shapely.ops import unary_union


# =========================================================
# BRUNSWICK CREEK V10742
# AINSLIE CREEK V10755
#
# DAILY NASA FIRMS AUTOMATION
#
# Background: Streets
# FIRMS: Simple Mode
# =========================================================


# =========================================================
# FOLDERS
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
FRAME_DIR = ROOT / "frames"
OUTPUT_DIR = ROOT / "output"
STATUS_DIR = ROOT / "status"

for folder in (
    DATA_DIR,
    FRAME_DIR,
    OUTPUT_DIR,
    STATUS_DIR
):
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# NASA FIRMS MAP KEY
#
# Loaded from GitHub Secret:
# FIRMS_MAP_KEY
# =========================================================

FIRMS_MAP_KEY = os.environ.get(
    "FIRMS_MAP_KEY",
    ""
).strip()


if not FIRMS_MAP_KEY:

    raise RuntimeError(
        "Missing GitHub secret FIRMS_MAP_KEY."
    )


# =========================================================
# TIME / DATES
# =========================================================

TZ = ZoneInfo(
    "America/Vancouver"
)

TODAY = datetime.now(
    TZ
).date()


START_DATE = date(
    2026,
    7,
    1
)


# Do not use today's incomplete satellite data
# when deciding whether the fire has stopped expanding.

EVALUATION_END = (
    TODAY
    - timedelta(days=1)
)


# =========================================================
# MAP AREA
#
# Brunswick Creek + Ainslie Creek
#
# west,south,east,north
# =========================================================

BBOX = (
    -121.95,
    49.55,
    -120.95,
    50.45
)


# =========================================================
# FIRMS SOURCE
# =========================================================

SOURCE = "VIIRS_SNPP_NRT"


# =========================================================
# NASA FIRMS SERVERS
#
# Try the secondary server FIRST.
#
# This helps because GitHub Actions had trouble
# reaching the primary NASA server.
# =========================================================

FIRMS_HOSTS = [

    "https://firms2.modaps.eosdis.nasa.gov",

    "https://firms.modaps.eosdis.nasa.gov",
]


# =========================================================
# DAILY UPDATE SETTINGS
# =========================================================

# Re-check recent days in case NASA adds
# late satellite detections.

REFRESH_DAYS = 7


# =========================================================
# AUTO-STOP SETTINGS
#
# Stop the automatic workflow after
# 5 completed days without meaningful expansion.
#
# A new point must extend at least 500 metres
# beyond the earlier accumulated hotspot footprint.
# =========================================================

NO_EXPANSION_DAYS_TO_STOP = 5

EXPANSION_TOLERANCE_METERS = 500


# =========================================================
# VIDEO SETTINGS
# =========================================================

VIDEO_WIDTH = 960

VIDEO_HEIGHT = 864

FPS = 3

MP4_FINAL_PAUSE_SECONDS = 5


# =========================================================
# MAP LABELS
# =========================================================

PLACES = {

    "Boston Bar":
        (-121.44, 49.86),

    "North Bend":
        (-121.45, 49.88),

    "Hells Gate":
        (-121.42, 49.78),

    "Nahatlatch":
        (-121.72, 49.96),

    "Lytton":
        (-121.58, 50.23),
}


HIGHWAY_1_LABEL = (
    -121.43,
    49.92
)


# =========================================================
# SAVED FILES
# =========================================================

CSV_CACHE = (
    DATA_DIR
    / "firms_data.csv"
)

BASEMAP_FILE = (
    DATA_DIR
    / "streets_basemap.png"
)

BASEMAP_EXTENT_FILE = (
    DATA_DIR
    / "streets_basemap_extent.json"
)

STATE_FILE = (
    STATUS_DIR
    / "state.json"
)

COMPLETE_FILE = (
    STATUS_DIR
    / "COMPLETE.txt"
)


MP4_OUT = (
    OUTPUT_DIR
    / "Brunswick_Ainslie_SIMPLE.mp4"
)

GIF_OUT = (
    OUTPUT_DIR
    / "Brunswick_Ainslie_SLIDESHOW_ONCE.gif"
)

HTML_OUT = (
    OUTPUT_DIR
    / "Brunswick_Ainslie_SLIDESHOW.html"
)

LATEST_OUT = (
    OUTPUT_DIR
    / "Brunswick_Ainslie_LATEST.png"
)


# =========================================================
# PRINT HELPER
# =========================================================

def log(*items):

    print(
        *items,
        flush=True
    )


# =========================================================
# DATE HELPERS
# =========================================================

def daterange(
    start,
    end
):

    day = start

    while day <= end:

        yield day

        day += timedelta(
            days=1
        )


def chunked(
    start,
    end,
    number=5
):

    day = start

    while day <= end:

        chunk_end = min(

            day
            + timedelta(
                days=number - 1
            ),

            end
        )


        days = (
            chunk_end - day
        ).days + 1


        yield (
            day,
            days
        )


        day = (
            chunk_end
            + timedelta(days=1)
        )


# =========================================================
# CLEAN FIRMS DATA
# =========================================================

def normalize(
    dataframe
):

    if dataframe.empty:

        return dataframe.copy()


    dataframe = dataframe.copy()


    dataframe.columns = [

        str(column).lower()

        for column
        in dataframe.columns
    ]


    required = {

        "acq_date",
        "latitude",
        "longitude"
    }


    if not required.issubset(
        dataframe.columns
    ):

        missing = (
            required
            - set(
                dataframe.columns
            )
        )

        raise ValueError(

            "FIRMS response missing "
            f"{sorted(missing)}"
        )


    dataframe["acq_date"] = (

        pd.to_datetime(

            dataframe["acq_date"],

            errors="coerce"

        ).dt.date
    )


    dataframe["latitude"] = (

        pd.to_numeric(

            dataframe["latitude"],

            errors="coerce"
        )
    )


    dataframe["longitude"] = (

        pd.to_numeric(

            dataframe["longitude"],

            errors="coerce"
        )
    )


    if "frp" in dataframe.columns:

        dataframe["frp"] = (

            pd.to_numeric(

                dataframe["frp"],

                errors="coerce"
            )
        )

    else:

        dataframe["frp"] = np.nan


    if "acq_time" in dataframe.columns:

        dataframe["acq_time"] = (

            dataframe["acq_time"]

            .astype(str)

            .str.replace(
                r"\.0$",
                "",
                regex=True
            )

            .str.zfill(4)
        )


    dataframe = dataframe.dropna(

        subset=[

            "acq_date",

            "latitude",

            "longitude"
        ]
    )


    duplicate_keys = [

        column

        for column in (

            "latitude",

            "longitude",

            "acq_date",

            "acq_time"
        )

        if column
        in dataframe.columns
    ]


    if duplicate_keys:

        dataframe = (
            dataframe.drop_duplicates(
                subset=duplicate_keys
            )
        )


    return dataframe.reset_index(
        drop=True
    )


# =========================================================
# LOAD SAVED FIRMS DATA
# =========================================================

def load_saved():

    if not CSV_CACHE.exists():

        return pd.DataFrame()


    dataframe = pd.read_csv(
        CSV_CACHE
    )


    return normalize(
        dataframe
    )


# =========================================================
# DOWNLOAD FIRMS DATA
#
# Tries NASA secondary server first.
# If that fails, tries the primary server.
# =========================================================

def fetch_range(
    start,
    end
):

    if start > end:

        return (
            pd.DataFrame(),
            True
        )


    west, south, east, north = BBOX


    bbox = (

        f"{west},"
        f"{south},"
        f"{east},"
        f"{north}"
    )


    pieces = []

    all_ok = True


    for chunk_start, days in chunked(
        start,
        end,
        5
    ):

        success = False


        log()
        log(
            "===================================="
        )

        log(
            "FIRMS DATE:",
            chunk_start
        )

        log(
            "Days requested:",
            days
        )


        # =================================================
        # TRY NASA SERVERS
        # =================================================

        for host in FIRMS_HOSTS:

            log()
            log(
                "Trying NASA FIRMS server:"
            )

            log(
                host
            )


            url = (

                f"{host}/api/area/csv/"

                f"{FIRMS_MAP_KEY}/"

                f"{SOURCE}/"

                f"{bbox}/"

                f"{days}/"

                f"{chunk_start.isoformat()}"
            )


            try:

                response = requests.get(

                    url,

                    timeout=30,

                    headers={

                        "User-Agent":
                        "Brunswick-Ainslie-FIRMS-GitHub-Automation/1.0"

                    }
                )


                response.raise_for_status()


                text = (
                    response.text.strip()
                )


                # =========================================
                # EMPTY RESPONSE
                # =========================================

                if not text:

                    log(
                        "No FIRMS hotspots "
                        "returned for this period."
                    )

                    success = True

                    break


                # =========================================
                # REJECT HTML ERROR PAGES
                # =========================================

                if text.startswith("<"):

                    raise RuntimeError(

                        "NASA returned HTML "
                        "instead of CSV."
                    )


                # =========================================
                # READ CSV
                # =========================================

                try:

                    temp = pd.read_csv(
                        StringIO(text)
                    )

                except pd.errors.EmptyDataError:

                    temp = pd.DataFrame()


                if temp.empty:

                    log(
                        "Valid FIRMS response "
                        "but no detections."
                    )

                    success = True

                    break


                # =========================================
                # VERIFY FIRMS COLUMNS
                # =========================================

                actual_columns = {

                    str(column).lower()

                    for column
                    in temp.columns
                }


                required_columns = {

                    "acq_date",

                    "latitude",

                    "longitude"
                }


                if not required_columns.issubset(
                    actual_columns
                ):

                    raise RuntimeError(

                        "Unexpected FIRMS response. "
                        "Required hotspot columns "
                        "were not present."
                    )


                pieces.append(
                    temp
                )


                log(
                    "SUCCESS:"
                )

                log(
                    len(temp),
                    "detections downloaded"
                )


                log(
                    "Server used:",
                    host
                )


                success = True

                break


            except Exception as error:

                log()
                log(
                    "Server failed:"
                )

                log(
                    host
                )

                log(
                    type(error).__name__,
                    ":",
                    error
                )


        # =================================================
        # BOTH NASA SERVERS FAILED
        # =================================================

        if not success:

            all_ok = False

            log()
            log(
                "⚠️ BOTH NASA FIRMS SERVERS "
                "FAILED FOR:"
            )

            log(
                chunk_start
            )


    # =====================================================
    # NO RECORDS RECEIVED
    # =====================================================

    if not pieces:

        return (
            pd.DataFrame(),
            all_ok
        )


    combined = pd.concat(

        pieces,

        ignore_index=True
    )


    return (
        normalize(
            combined
        ),
        all_ok
    )


# =========================================================
# UPDATE SAVED FIRMS DATA
# =========================================================

def update_data():

    old_data = load_saved()


    if old_data.empty:

        refresh_start = START_DATE

    else:

        refresh_start = max(

            START_DATE,

            TODAY
            - timedelta(
                days=REFRESH_DAYS - 1
            )
        )


    log()
    log(
        "Refreshing FIRMS:"
    )

    log(
        refresh_start,
        "through",
        TODAY
    )


    fresh_data, refresh_ok = (
        fetch_range(

            refresh_start,

            TODAY
        )
    )


    if (
        old_data.empty
        and fresh_data.empty
    ):

        raise RuntimeError(

            "No FIRMS data retrieved. "
            "Check the NASA FIRMS messages "
            "above in this Actions log."
        )


    if old_data.empty:

        merged = fresh_data


    elif fresh_data.empty:

        merged = old_data


    else:

        merged = pd.concat(

            [
                old_data,
                fresh_data
            ],

            ignore_index=True
        )


    merged = normalize(
        merged
    )


    merged = merged[

        (
            merged["acq_date"]
            >= START_DATE
        )

        &

        (
            merged["acq_date"]
            <= TODAY
        )

    ].copy()


    merged.to_csv(

        CSV_CACHE,

        index=False
    )


    log()
    log(
        "Saved hotspot detections:",
        len(merged)
    )


    return (

        merged,

        refresh_start,

        refresh_ok
    )


# =========================================================
# CONVERT POINTS TO WEB MERCATOR
# =========================================================

def to3857(
    dataframe
):

    if dataframe.empty:

        return gpd.GeoDataFrame(

            dataframe.copy(),

            geometry=[],

            crs="EPSG:4326"
        )


    geodata = gpd.GeoDataFrame(

        dataframe.copy(),

        geometry=[

            Point(
                longitude,
                latitude
            )

            for longitude, latitude
            in zip(

                dataframe["longitude"],

                dataframe["latitude"]
            )
        ],

        crs="EPSG:4326"
    )


    return geodata.to_crs(
        epsg=3857
    )


# =========================================================
# MAP BOUNDS
# =========================================================

def bounds3857():

    west, south, east, north = BBOX


    geodata = gpd.GeoDataFrame(

        geometry=[

            Point(
                west,
                south
            ),

            Point(
                east,
                north
            )

        ],

        crs="EPSG:4326"

    ).to_crs(
        epsg=3857
    )


    return (

        float(
            geodata.geometry.x.min()
        ),

        float(
            geodata.geometry.y.min()
        ),

        float(
            geodata.geometry.x.max()
        ),

        float(
            geodata.geometry.y.max()
        )
    )


# =========================================================
# STREET BASEMAP
# =========================================================

def basemap():

    bounds = bounds3857()


    (
        xmin,
        ymin,
        xmax,
        ymax
    ) = bounds


    # =====================================================
    # LOAD SAVED MAP
    # =====================================================

    if (
        BASEMAP_FILE.exists()

        and

        BASEMAP_EXTENT_FILE.exists()
    ):

        log(
            "Loading saved Streets basemap."
        )


        image = np.array(

            Image.open(
                BASEMAP_FILE
            ).convert(
                "RGBA"
            )
        )


        extent = json.loads(

            BASEMAP_EXTENT_FILE.read_text()
        )


        return (
            image,
            extent,
            bounds
        )


    # =====================================================
    # DOWNLOAD MAP
    # =====================================================

    log(
        "Downloading Streets basemap once..."
    )


    image, extent = ctx.bounds2img(

        xmin,
        ymin,
        xmax,
        ymax,

        zoom=9,

        source=(
            ctx.providers
            .Esri
            .WorldStreetMap
        )
    )


    Image.fromarray(
        image
    ).save(
        BASEMAP_FILE,
        optimize=True
    )


    BASEMAP_EXTENT_FILE.write_text(

        json.dumps(

            [
                float(value)
                for value in extent
            ]
        )
    )


    return (
        image,
        extent,
        bounds
    )


# =========================================================
# MAP LABELS
# =========================================================

def labels():

    places_dataframe = pd.DataFrame(

        [

            {

                "name":
                    name,

                "longitude":
                    longitude,

                "latitude":
                    latitude
            }

            for name,
            (
                longitude,
                latitude
            )

            in PLACES.items()
        ]
    )


    places = gpd.GeoDataFrame(

        places_dataframe,

        geometry=[

            Point(
                longitude,
                latitude
            )

            for longitude, latitude
            in zip(

                places_dataframe["longitude"],

                places_dataframe["latitude"]
            )
        ],

        crs="EPSG:4326"

    ).to_crs(
        epsg=3857
    )


    highway = gpd.GeoSeries(

        [

            Point(
                *HIGHWAY_1_LABEL
            )

        ],

        crs="EPSG:4326"

    ).to_crs(
        epsg=3857
    ).iloc[0]


    return (
        places,
        highway
    )


# =========================================================
# FRAME PATH
# =========================================================

def frame_path(
    day
):

    return (

        FRAME_DIR

        / (
            "simple_frame_"
            f"{day.isoformat()}.png"
        )
    )


# =========================================================
# CREATE ONE DAILY FRAME
# =========================================================

def create_frame(

    day,

    data,

    background,

    background_extent,

    bounds,

    places,

    highway
):


    visible = data[

        (
            data["acq_date"]
            >= START_DATE
        )

        &

        (
            data["acq_date"]
            <= day
        )
    ]


    hotspots = to3857(
        visible
    )


    (
        xmin,
        ymin,
        xmax,
        ymax
    ) = bounds


    # Fixed 960 x 864 image

    fig, axis = plt.subplots(

        figsize=(
            9.6,
            8.64
        ),

        dpi=100
    )


    axis.set_xlim(
        xmin,
        xmax
    )

    axis.set_ylim(
        ymin,
        ymax
    )


    # =====================================================
    # STREETS BACKGROUND
    # =====================================================

    axis.imshow(

        background,

        extent=background_extent,

        interpolation="bilinear",

        zorder=1
    )


    # =====================================================
    # FIRMS SIMPLE MODE
    #
    # All accumulated hotspots use
    # the same solid red style.
    # =====================================================

    if not hotspots.empty:

        axis.scatter(

            hotspots.geometry.x,

            hotspots.geometry.y,

            s=26,

            color="red",

            alpha=0.92,

            edgecolors="black",

            linewidths=0.35,

            zorder=5
        )


    # =====================================================
    # PLACE LABELS
    # =====================================================

    for _, row in places.iterrows():

        x = row.geometry.x

        y = row.geometry.y


        axis.scatter(

            x,
            y,

            s=22,

            color="black",

            zorder=7
        )


        axis.annotate(

            row["name"],

            (
                x,
                y
            ),

            xytext=(
                5,
                5
            ),

            textcoords="offset points",

            fontsize=9,

            weight="bold",

            bbox=dict(

                facecolor="white",

                alpha=0.80,

                edgecolor="none",

                pad=1.2
            ),

            zorder=8
        )


    # =====================================================
    # HIGHWAY 1
    # =====================================================

    axis.annotate(

        "HIGHWAY 1",

        (
            highway.x,
            highway.y
        ),

        fontsize=10,

        weight="bold",

        rotation=70,

        ha="center",

        bbox=dict(

            facecolor="white",

            alpha=0.85,

            edgecolor="black",

            pad=2
        ),

        zorder=9
    )


    # =====================================================
    # TITLE
    # =====================================================

        # Main title
    fig.suptitle(
        "Brunswick Creek V10742 + Ainslie Creek V10755\n"
        "Fire/Hotspot Progression",
        fontsize=15,
        weight="bold",
        y=0.985
    )

    # Large dark-blue date
    fig.text(
        0.5,
        0.895,
        day.strftime("%B %d, %Y"),
        ha="center",
        va="center",
        fontsize=26,
        weight="bold",
        color="darkblue"
    )


    # =====================================================
    # FOOTER
    # =====================================================

    fig.text(

        0.5,
        0.015,

        "Background: Streets  |  "
        "Fire/Hotspots: NASA FIRMS VIIRS  |  "
        "Hotspots are not an official fire perimeter",

        ha="center",

        fontsize=7.5
    )


    axis.set_axis_off()


    fig.subplots_adjust(

        left=0.015,

        right=0.985,

        bottom=0.055,

        top=0.82
    )


    path = frame_path(
        day
    )


    fig.savefig(

        path,

        dpi=100,

        facecolor="white"
    )


    plt.close(
        fig
    )


    log(

        "Frame",

        day,

        "| accumulated hotspots:",

        len(visible)
    )


    return path


# =========================================================
# BUILD / UPDATE FRAMES
# =========================================================

def build_frames(

    data,

    refresh_start,

    background,

    background_extent,

    bounds,

    places,

    highway
):


    existing_frame = next(

        FRAME_DIR.glob(
            "simple_frame_*.png"
        ),

        None
    )


    if existing_frame is None:

        regenerate_from = START_DATE

    else:

        regenerate_from = refresh_start


    log()
    log(
        "Regenerating frames from:",
        regenerate_from
    )


    for day in daterange(

        regenerate_from,

        TODAY
    ):

        create_frame(

            day,

            data,

            background,

            background_extent,

            bounds,

            places,

            highway
        )


    files = []


    for day in daterange(

        START_DATE,

        TODAY
    ):

        path = frame_path(
            day
        )


        if not path.exists():

            create_frame(

                day,

                data,

                background,

                background_extent,

                bounds,

                places,

                highway
            )


        files.append(
            path
        )


    return files


# =========================================================
# FORCE VIDEO FRAME SIZE
# =========================================================

def fixed_frame(

    path,

    width=VIDEO_WIDTH,

    height=VIDEO_HEIGHT
):


    image = Image.open(
        path
    ).convert(
        "RGB"
    )


    image.thumbnail(

        (
            width,
            height
        ),

        Image.Resampling.LANCZOS
    )


    canvas = Image.new(

        "RGB",

        (
            width,
            height
        ),

        "white"
    )


    canvas.paste(

        image,

        (

            (
                width
                - image.width
            ) // 2,

            (
                height
                - image.height
            ) // 2
        )
    )


    return np.array(
        canvas
    )


# =========================================================
# BUILD MP4
# =========================================================

def build_mp4(
    files
):

    log()
    log(
        "Building MP4..."
    )


    writer = imageio.get_writer(

        MP4_OUT,

        fps=FPS,

        codec="libx264",

        quality=7,

        macro_block_size=16
    )


    try:

        for number, path in enumerate(

            files,

            start=1
        ):

            log(

                "MP4 frame",

                number,

                "of",

                len(files)
            )


            writer.append_data(

                fixed_frame(
                    path
                )
            )


        # Hold final MP4 frame for 5 seconds.

        last = fixed_frame(
            files[-1]
        )


        for _ in range(

            FPS
            * MP4_FINAL_PAUSE_SECONDS
        ):

            writer.append_data(
                last
            )


    finally:

        writer.close()


# =========================================================
# BUILD ONE-PLAY GIF
# =========================================================

def build_gif(
    files
):

    log()
    log(
        "Building one-play GIF..."
    )


    images = []


    for path in files:

        image = Image.open(
            path
        ).convert(
            "RGB"
        )


        image.thumbnail(

            (
                640,
                576
            ),

            Image.Resampling.LANCZOS
        )


        canvas = Image.new(

            "RGB",

            (
                640,
                576
            ),

            "white"
        )


        canvas.paste(

            image,

            (

                (
                    640
                    - image.width
                ) // 2,

                (
                    576
                    - image.height
                ) // 2
            )
        )


        images.append(

            canvas.convert(

                "P",

                palette=(
                    Image.Palette.ADAPTIVE
                )
            )
        )


    images[0].save(

        GIF_OUT,

        save_all=True,

        append_images=images[1:],

        duration=int(
            1000 / FPS
        ),

        optimize=True,

        disposal=1

        # No loop command:
        # most viewers play it once.
    )


# =========================================================
# CREATE IMAGE DATA FOR HTML SLIDESHOW
# =========================================================

def data_uri(
    path
):

    image = Image.open(
        path
    ).convert(
        "RGB"
    )


    image.thumbnail(

        (
            640,
            576
        ),

        Image.Resampling.LANCZOS
    )


    canvas = Image.new(

        "RGB",

        (
            640,
            576
        ),

        "white"
    )


    canvas.paste(

        image,

        (

            (
                640
                - image.width
            ) // 2,

            (
                576
                - image.height
            ) // 2
        )
    )


    buffer = io.BytesIO()


    canvas.save(

        buffer,

        format="JPEG",

        quality=74,

        optimize=True
    )


    encoded = base64.b64encode(

        buffer.getvalue()

    ).decode()


    return (

        "data:image/jpeg;base64,"

        + encoded
    )


# =========================================================
# HTML SLIDESHOW
#
# Plays once and stays on final frame
# until user closes the screen.
# =========================================================

def build_html(
    files
):

    log()
    log(
        "Building standalone slideshow..."
    )


    images = [

        data_uri(path)

        for path
        in files
    ]


    dates = [

        path.stem.replace(

            "simple_frame_",

            ""
        )

        for path
        in files
    ]


    html = """
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width,initial-scale=1"
>

<title>
Brunswick + Ainslie Slideshow
</title>

<style>

body {

    margin: 0;

    background: #111;

    color: #fff;

    font-family: Arial;

    display: grid;

    min-height: 100vh;

    place-items: center;
}

.w {

    width: min(96vw,1000px);

    text-align: center;
}

img {

    width: 100%;

    height: auto;

    background: #fff;
}

.r {

    display: flex;

    gap: 12px;

    justify-content: space-between;

    align-items: center;

    margin-top: 10px;
}

button {

    font: inherit;

    padding: 10px 16px;
}

</style>

</head>


<body>


<div class="w">

<img
    id="s"
    alt="Wildfire hotspot slideshow"
>


<div class="r">

<div id="t"></div>

<button id="b">
Replay
</button>

</div>

</div>


<script>

const f = IMAGES;

const d = DATES;

const s =
document.getElementById("s");

const t =
document.getElementById("t");

const b =
document.getElementById("b");


let i = 0;

let x = null;


function show() {

    s.src = f[i];

    t.textContent =
        d[i]
        + " • frame "
        + (i + 1)
        + " of "
        + f.length;
}


function start() {

    if (x) {

        clearInterval(x);
    }


    i = 0;

    show();


    x = setInterval(

        () => {

            if (
                i
                < f.length - 1
            ) {

                i++;

                show();

            }

            else {

                clearInterval(x);

                x = null;


                t.textContent =

                    d[i]

                    + " • FINAL FRAME — "
                    + "remains until you "
                    + "close the screen";
            }

        },

        DELAY
    );
}


b.addEventListener(

    "click",

    start
);


start();

</script>


</body>

</html>
"""


    html = html.replace(

        "IMAGES",

        json.dumps(
            images
        )
    )


    html = html.replace(

        "DATES",

        json.dumps(
            dates
        )
    )


    html = html.replace(

        "DELAY",

        str(
            int(
                1000 / FPS
            )
        )
    )


    HTML_OUT.write_text(

        html,

        encoding="utf-8"
    )


# =========================================================
# BUILD ALL OUTPUTS
# =========================================================

def build_outputs(
    files
):

    shutil.copy2(

        files[-1],

        LATEST_OUT
    )


    build_mp4(
        files
    )


    build_gif(
        files
    )


    build_html(
        files
    )


# =========================================================
# CHECK WHETHER FIRE EXPANDED
# =========================================================

def expanded_on_day(

    data,

    day
):


    previous = data[

        data["acq_date"]
        < day
    ]


    current = data[

        data["acq_date"]
        == day
    ]


    if current.empty:

        return False


    if previous.empty:

        return True


    previous_geo = to3857(
        previous
    )


    current_geo = to3857(
        current
    )


    footprint = unary_union(

        [

            geometry.buffer(
                EXPANSION_TOLERANCE_METERS
            )

            for geometry
            in previous_geo.geometry
        ]
    )


    return any(

        not footprint.covers(
            point
        )

        for point
        in current_geo.geometry
    )


# =========================================================
# COUNT DAYS WITHOUT EXPANSION
# =========================================================

def no_expansion_days(

    data,

    end
):


    if end < START_DATE:

        return 0


    number = 0

    day = end


    while day >= START_DATE:

        if expanded_on_day(

            data,

            day
        ):

            break


        number += 1


        day -= timedelta(
            days=1
        )


    return number


# =========================================================
# SAVE STATUS
# =========================================================

def save_state(

    data,

    refresh_ok,

    number,

    complete
):


    state = {

        "last_run_local":

            datetime.now(
                TZ
            ).isoformat(),


        "today":

            TODAY.isoformat(),


        "evaluation_through":

            EVALUATION_END.isoformat(),


        "firms_refresh_success":

            bool(
                refresh_ok
            ),


        "hotspot_detections":

            int(
                len(data)
            ),


        "no_expansion_days":

            int(
                number
            ),


        "stop_after_days":

            NO_EXPANSION_DAYS_TO_STOP,


        "expansion_tolerance_meters":

            EXPANSION_TOLERANCE_METERS,


        "complete":

            bool(
                complete
            )
    }


    STATE_FILE.write_text(

        json.dumps(

            state,

            indent=2
        )
    )


# =========================================================
# CREATE COMPLETE MARKER
# =========================================================

def mark_complete(
    number
):


    COMPLETE_FILE.write_text(

        "Brunswick Creek V10742 + "
        "Ainslie Creek V10755 "
        "animation marked complete.\n\n"

        f"No meaningful expansion for "
        f"{number} consecutive completed "
        f"days through {EVALUATION_END}.\n"

        f"Expansion rule: new hotspot > "
        f"{EXPANSION_TOLERANCE_METERS} m "
        "beyond prior accumulated hotspot "
        "footprint.\n\n"

        "This is an animation stop rule "
        "based on FIRMS satellite hotspots, "
        "NOT an official wildfire status.\n"
    )


# =========================================================
# MAIN DAILY RUN
# =========================================================

def main():

    log()
    log(
        "============================================"
    )

    log(
        "BRUNSWICK + AINSLIE DAILY ANIMATION"
    )

    log(
        "Local date:",
        TODAY
    )

    log(
        "============================================"
    )


    # =====================================================
    # ALREADY COMPLETE
    # =====================================================

    if COMPLETE_FILE.exists():

        log(
            "Already COMPLETE."
        )

        log(
            "Delete status/COMPLETE.txt "
            "to resume."
        )

        return


    # =====================================================
    # DOWNLOAD / UPDATE FIRMS
    # =====================================================

    (
        data,

        refresh_start,

        refresh_ok

    ) = update_data()


    # =====================================================
    # MAP
    # =====================================================

    (
        background,

        background_extent,

        bounds

    ) = basemap()


    (
        places,

        highway

    ) = labels()


    # =====================================================
    # FRAMES
    # =====================================================

    files = build_frames(

        data,

        refresh_start,

        background,

        background_extent,

        bounds,

        places,

        highway
    )


    # =====================================================
    # MP4 / GIF / HTML / LATEST PNG
    # =====================================================

    build_outputs(
        files
    )


    # =====================================================
    # EXPANSION CHECK
    # =====================================================

    number = no_expansion_days(

        data,

        EVALUATION_END
    )


    complete = (

        refresh_ok

        and

        number
        >= NO_EXPANSION_DAYS_TO_STOP
    )


    save_state(

        data,

        refresh_ok,

        number,

        complete
    )


    log()
    log(
        "Completed days without "
        "meaningful expansion:",
        number
    )


    # Never stop workflow if NASA download
    # had an error.

    if not refresh_ok:

        log(
            "FIRMS refresh had an error."
        )

        log(
            "Automatic stop suppressed "
            "for safety."
        )


    if complete:

        mark_complete(
            number
        )

        log()
        log(
            "✅ COMPLETE"
        )

        log(
            "Workflow will disable itself "
            "after publishing outputs."
        )

    else:

        log()
        log(
            "🔥 ACTIVE"
        )

        log(
            "Daily updates will continue."
        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()
