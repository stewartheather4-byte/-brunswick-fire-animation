#!/usr/bin/env python3
from __future__ import annotations

import base64, io, json, os, shutil
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import contextily as ctx
import geopandas as gpd
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from shapely.geometry import Point
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent
DATA_DIR, FRAME_DIR, OUTPUT_DIR, STATUS_DIR = [ROOT / x for x in ('data','frames','output','status')]
for d in (DATA_DIR, FRAME_DIR, OUTPUT_DIR, STATUS_DIR): d.mkdir(parents=True, exist_ok=True)

FIRMS_MAP_KEY = os.environ.get('FIRMS_MAP_KEY','').strip()
if not FIRMS_MAP_KEY:
    raise RuntimeError('Missing GitHub secret FIRMS_MAP_KEY.')

TZ = ZoneInfo('America/Vancouver')
TODAY = datetime.now(TZ).date()
START_DATE = date(2026, 7, 1)
BBOX = (-121.95, 49.55, -120.95, 50.45)  # west,south,east,north
SOURCE = 'VIIRS_SNPP_NRT'
REFRESH_DAYS = 7
EVALUATION_END = TODAY - timedelta(days=1)
NO_EXPANSION_DAYS_TO_STOP = 5
EXPANSION_TOLERANCE_METERS = 500
VIDEO_WIDTH, VIDEO_HEIGHT = 960, 864
FPS = 3
MP4_FINAL_PAUSE_SECONDS = 5

PLACES = {
    'Boston Bar': (-121.44, 49.86),
    'North Bend': (-121.45, 49.88),
    'Hells Gate': (-121.42, 49.78),
    'Nahatlatch': (-121.72, 49.96),
    'Lytton': (-121.58, 50.23),
}
HIGHWAY_1_LABEL = (-121.43, 49.92)

CSV_CACHE = DATA_DIR/'firms_data.csv'
BASEMAP_FILE = DATA_DIR/'streets_basemap.png'
BASEMAP_EXTENT_FILE = DATA_DIR/'streets_basemap_extent.json'
STATE_FILE = STATUS_DIR/'state.json'
COMPLETE_FILE = STATUS_DIR/'COMPLETE.txt'
MP4_OUT = OUTPUT_DIR/'Brunswick_Ainslie_SIMPLE.mp4'
GIF_OUT = OUTPUT_DIR/'Brunswick_Ainslie_SLIDESHOW_ONCE.gif'
HTML_OUT = OUTPUT_DIR/'Brunswick_Ainslie_SLIDESHOW.html'
LATEST_OUT = OUTPUT_DIR/'Brunswick_Ainslie_LATEST.png'


def log(*x): print(*x, flush=True)

def daterange(start, end):
    d=start
    while d<=end:
        yield d
        d += timedelta(days=1)

def chunked(start,end,n=5):
    d=start
    while d<=end:
        e=min(d+timedelta(days=n-1),end)
        yield d,(e-d).days+1
        d=e+timedelta(days=1)

def normalize(df):
    if df.empty: return df.copy()
    df=df.copy(); df.columns=[str(c).lower() for c in df.columns]
    need={'acq_date','latitude','longitude'}
    if not need.issubset(df.columns):
        raise ValueError(f'FIRMS response missing {sorted(need-set(df.columns))}')
    df['acq_date']=pd.to_datetime(df['acq_date'],errors='coerce').dt.date
    df['latitude']=pd.to_numeric(df['latitude'],errors='coerce')
    df['longitude']=pd.to_numeric(df['longitude'],errors='coerce')
    df['frp']=pd.to_numeric(df['frp'],errors='coerce') if 'frp' in df else np.nan
    if 'acq_time' in df:
        df['acq_time']=df['acq_time'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(4)
    df=df.dropna(subset=['acq_date','latitude','longitude'])
    keys=[c for c in ('latitude','longitude','acq_date','acq_time') if c in df.columns]
    if keys: df=df.drop_duplicates(subset=keys)
    return df.reset_index(drop=True)

def load_saved():
    return normalize(pd.read_csv(CSV_CACHE)) if CSV_CACHE.exists() else pd.DataFrame()

def fetch_range(start,end):
    if start>end: return pd.DataFrame(), True
    west,south,east,north=BBOX
    bbox=f'{west},{south},{east},{north}'
    pieces=[]; all_ok=True
    for chunk_start,days in chunked(start,end,5):
        url=('https://firms.modaps.eosdis.nasa.gov/api/area/csv/'
             f'{FIRMS_MAP_KEY}/{SOURCE}/{bbox}/{days}/{chunk_start.isoformat()}')
        log('FIRMS:',chunk_start,'for',days,'day(s)')
        try:
            r=requests.get(url,timeout=90); r.raise_for_status(); text=r.text.strip()
            if not text: continue
            if text.startswith('<'): raise RuntimeError('HTML response instead of CSV')
            try: tmp=pd.read_csv(StringIO(text))
            except pd.errors.EmptyDataError: tmp=pd.DataFrame()
            if tmp.empty: continue
            low={str(c).lower() for c in tmp.columns}
            if not {'acq_date','latitude','longitude'}.issubset(low):
                raise RuntimeError('Unexpected FIRMS response: '+text[:160].replace('\n',' '))
            pieces.append(tmp)
        except Exception as exc:
            all_ok=False; log('WARNING FIRMS request failed:',exc)
    if not pieces: return pd.DataFrame(), all_ok
    return normalize(pd.concat(pieces,ignore_index=True)), all_ok

def update_data():
    old=load_saved()
    refresh_start=START_DATE if old.empty else max(START_DATE,TODAY-timedelta(days=REFRESH_DAYS-1))
    log('Refreshing',refresh_start,'through',TODAY)
    fresh,ok=fetch_range(refresh_start,TODAY)
    if old.empty and fresh.empty:
        raise RuntimeError('No FIRMS data retrieved. Check key/log.')
    merged=fresh if old.empty else old if fresh.empty else pd.concat([old,fresh],ignore_index=True)
    merged=normalize(merged)
    merged=merged[(merged.acq_date>=START_DATE)&(merged.acq_date<=TODAY)].copy()
    merged.to_csv(CSV_CACHE,index=False)
    log('Saved detections:',len(merged))
    return merged,refresh_start,ok

def to3857(df):
    if df.empty: return gpd.GeoDataFrame(df.copy(),geometry=[],crs='EPSG:4326')
    g=gpd.GeoDataFrame(df.copy(),geometry=[Point(lon,lat) for lon,lat in zip(df.longitude,df.latitude)],crs='EPSG:4326')
    return g.to_crs(epsg=3857)

def bounds3857():
    w,s,e,n=BBOX
    g=gpd.GeoDataFrame(geometry=[Point(w,s),Point(e,n)],crs='EPSG:4326').to_crs(epsg=3857)
    return float(g.geometry.x.min()),float(g.geometry.y.min()),float(g.geometry.x.max()),float(g.geometry.y.max())

def basemap():
    bounds=bounds3857(); xmin,ymin,xmax,ymax=bounds
    if BASEMAP_FILE.exists() and BASEMAP_EXTENT_FILE.exists():
        return np.array(Image.open(BASEMAP_FILE).convert('RGBA')),json.loads(BASEMAP_EXTENT_FILE.read_text()),bounds
    log('Downloading Streets basemap once...')
    img,extent=ctx.bounds2img(xmin,ymin,xmax,ymax,zoom=9,source=ctx.providers.Esri.WorldStreetMap)
    Image.fromarray(img).save(BASEMAP_FILE,optimize=True)
    BASEMAP_EXTENT_FILE.write_text(json.dumps([float(x) for x in extent]))
    return img,extent,bounds

def labels():
    p=pd.DataFrame([{'name':n,'longitude':lon,'latitude':lat} for n,(lon,lat) in PLACES.items()])
    pg=gpd.GeoDataFrame(p,geometry=[Point(lon,lat) for lon,lat in zip(p.longitude,p.latitude)],crs='EPSG:4326').to_crs(epsg=3857)
    h=gpd.GeoSeries([Point(*HIGHWAY_1_LABEL)],crs='EPSG:4326').to_crs(epsg=3857).iloc[0]
    return pg,h

def frame_path(day): return FRAME_DIR/f'simple_frame_{day.isoformat()}.png'

def create_frame(day,data,bg,bg_extent,bounds,places,hwy):
    visible=data[(data.acq_date>=START_DATE)&(data.acq_date<=day)]
    hot=to3857(visible); xmin,ymin,xmax,ymax=bounds
    fig,ax=plt.subplots(figsize=(9.6,8.64),dpi=100)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    ax.imshow(bg,extent=bg_extent,interpolation='bilinear',zorder=1)
    if not hot.empty:
        ax.scatter(hot.geometry.x,hot.geometry.y,s=26,color='red',alpha=.92,edgecolors='black',linewidths=.35,zorder=5)
    for _,row in places.iterrows():
        x,y=row.geometry.x,row.geometry.y
        ax.scatter(x,y,s=22,color='black',zorder=7)
        ax.annotate(row['name'],(x,y),xytext=(5,5),textcoords='offset points',fontsize=9,weight='bold',
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1.2),zorder=8)
    ax.annotate('HIGHWAY 1',(hwy.x,hwy.y),fontsize=10,weight='bold',rotation=70,ha='center',
                bbox=dict(facecolor='white',alpha=.85,edgecolor='black',pad=2),zorder=9)
    fig.suptitle('Brunswick Creek V10742 + Ainslie Creek V10755\nNASA FIRMS Fire/Hotspot Progression\n'
                 +day.strftime('%B %d, %Y'),fontsize=15,weight='bold',y=.985)
    fig.text(.5,.015,'Background: Streets  |  Fire/Hotspots: NASA FIRMS VIIRS  |  Hotspots are not an official fire perimeter',
             ha='center',fontsize=7.5)
    ax.set_axis_off(); fig.subplots_adjust(left=.015,right=.985,bottom=.055,top=.82)
    path=frame_path(day); fig.savefig(path,dpi=100,facecolor='white'); plt.close(fig)
    log('Frame',day,'| accumulated hotspots:',len(visible)); return path

def build_frames(data,refresh_start,bg,bg_extent,bounds,places,hwy):
    any_frames=next(FRAME_DIR.glob('simple_frame_*.png'),None) is not None
    regen=refresh_start if any_frames else START_DATE
    log('Regenerating frames from',regen)
    for day in daterange(regen,TODAY): create_frame(day,data,bg,bg_extent,bounds,places,hwy)
    files=[]
    for day in daterange(START_DATE,TODAY):
        p=frame_path(day)
        if not p.exists(): create_frame(day,data,bg,bg_extent,bounds,places,hwy)
        files.append(p)
    return files

def fixed_frame(path,w=VIDEO_WIDTH,h=VIDEO_HEIGHT):
    im=Image.open(path).convert('RGB'); im.thumbnail((w,h),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(w,h),'white'); canvas.paste(im,((w-im.width)//2,(h-im.height)//2))
    return np.array(canvas)

def build_mp4(files):
    log('Building MP4...')
    wr=imageio.get_writer(MP4_OUT,fps=FPS,codec='libx264',quality=7,macro_block_size=16)
    try:
        for i,p in enumerate(files,1): log('MP4',i,'/',len(files)); wr.append_data(fixed_frame(p))
        last=fixed_frame(files[-1])
        for _ in range(FPS*MP4_FINAL_PAUSE_SECONDS): wr.append_data(last)
    finally: wr.close()

def build_gif(files):
    log('Building one-play GIF...'); images=[]
    for p in files:
        im=Image.open(p).convert('RGB'); im.thumbnail((640,576),Image.Resampling.LANCZOS)
        c=Image.new('RGB',(640,576),'white'); c.paste(im,((640-im.width)//2,(576-im.height)//2))
        images.append(c.convert('P',palette=Image.Palette.ADAPTIVE))
    images[0].save(GIF_OUT,save_all=True,append_images=images[1:],duration=int(1000/FPS),optimize=True,disposal=1)

def data_uri(path):
    im=Image.open(path).convert('RGB'); im.thumbnail((640,576),Image.Resampling.LANCZOS)
    c=Image.new('RGB',(640,576),'white'); c.paste(im,((640-im.width)//2,(576-im.height)//2))
    b=io.BytesIO(); c.save(b,format='JPEG',quality=74,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()

def build_html(files):
    log('Building standalone slideshow...')
    imgs=[data_uri(p) for p in files]; dates=[p.stem.replace('simple_frame_','') for p in files]
    html='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Brunswick + Ainslie Slideshow</title><style>body{margin:0;background:#111;color:#fff;font-family:Arial;display:grid;min-height:100vh;place-items:center}.w{width:min(96vw,1000px);text-align:center}img{width:100%;height:auto;background:#fff}.r{display:flex;gap:12px;justify-content:space-between;align-items:center;margin-top:10px}button{font:inherit;padding:10px 16px}</style></head><body><div class="w"><img id="s" alt="Wildfire hotspot slideshow"><div class="r"><div id="t"></div><button id="b">Replay</button></div></div><script>const f=IMAGES,d=DATES,s=document.getElementById('s'),t=document.getElementById('t'),b=document.getElementById('b');let i=0,x=null;function show(){s.src=f[i];t.textContent=d[i]+' • frame '+(i+1)+' of '+f.length}function start(){if(x)clearInterval(x);i=0;show();x=setInterval(()=>{if(i<f.length-1){i++;show()}else{clearInterval(x);x=null;t.textContent=d[i]+' • FINAL FRAME — remains until you close the screen'}},DELAY)}b.addEventListener('click',start);start();</script></body></html>'''
    html=html.replace('IMAGES',json.dumps(imgs)).replace('DATES',json.dumps(dates)).replace('DELAY',str(int(1000/FPS)))
    HTML_OUT.write_text(html,encoding='utf-8')

def build_outputs(files):
    shutil.copy2(files[-1],LATEST_OUT); build_mp4(files); build_gif(files); build_html(files)

def expanded_on_day(data,day):
    prev=data[data.acq_date<day]; cur=data[data.acq_date==day]
    if cur.empty: return False
    if prev.empty: return True
    pg=to3857(prev); cg=to3857(cur)
    footprint=unary_union([g.buffer(EXPANSION_TOLERANCE_METERS) for g in pg.geometry])
    return any(not footprint.covers(p) for p in cg.geometry)

def no_expansion_days(data,end):
    if end<START_DATE: return 0
    n=0; d=end
    while d>=START_DATE:
        if expanded_on_day(data,d): break
        n+=1; d-=timedelta(days=1)
    return n

def save_state(data,refresh_ok,n,complete):
    STATE_FILE.write_text(json.dumps({
        'last_run_local':datetime.now(TZ).isoformat(),'today':TODAY.isoformat(),'evaluation_through':EVALUATION_END.isoformat(),
        'firms_refresh_success':bool(refresh_ok),'hotspot_detections':int(len(data)),'no_expansion_days':int(n),
        'stop_after_days':NO_EXPANSION_DAYS_TO_STOP,'expansion_tolerance_meters':EXPANSION_TOLERANCE_METERS,'complete':bool(complete)
    },indent=2))

def mark_complete(n):
    COMPLETE_FILE.write_text(
        f'Brunswick Creek V10742 + Ainslie Creek V10755 animation marked complete.\n\n'
        f'No meaningful expansion for {n} consecutive completed days through {EVALUATION_END}.\n'
        f'Expansion rule: new hotspot > {EXPANSION_TOLERANCE_METERS} m beyond prior accumulated hotspot footprint.\n\n'
        'This is an animation stop rule based on FIRMS satellite hotspots, NOT an official wildfire status.\n')

def main():
    log('='*60); log('BRUNSWICK + AINSLIE DAILY ANIMATION',TODAY); log('='*60)
    if COMPLETE_FILE.exists():
        log('Already COMPLETE. Delete status/COMPLETE.txt to resume.'); return
    data,refresh_start,refresh_ok=update_data()
    bg,bg_extent,bounds=basemap(); places,hwy=labels()
    files=build_frames(data,refresh_start,bg,bg_extent,bounds,places,hwy)
    build_outputs(files)
    n=no_expansion_days(data,EVALUATION_END)
    complete=refresh_ok and n>=NO_EXPANSION_DAYS_TO_STOP
    save_state(data,refresh_ok,n,complete)
    log('Completed days without meaningful expansion:',n)
    if not refresh_ok: log('Refresh had an error; auto-stop suppressed for safety.')
    if complete:
        mark_complete(n); log('COMPLETE — workflow will disable itself after publishing outputs.')
    else: log('ACTIVE — daily updates continue.')

if __name__=='__main__': main()
