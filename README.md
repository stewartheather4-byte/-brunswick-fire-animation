# Brunswick Creek V10742 + Ainslie Creek V10755

Free daily NASA FIRMS wildfire/hotspot animation using GitHub Actions.

## What it does

- Runs automatically every day at **7:17 AM America/Vancouver**.
- Uses a **Streets** basemap.
- Uses FIRMS **Simple-style** display: accumulated hotspots are solid red.
- Starts on **July 1, 2026**.
- Saves one fixed-size frame per day.
- Re-checks the latest 7 days to catch late FIRMS detections.
- Publishes current MP4, one-play GIF, standalone HTML slideshow, and latest PNG in the repository's **Releases** area.
- The HTML slideshow plays once and then **stays on the final frame until the browser screen/tab is closed**.
- After **5 completed days** with no meaningful expansion, it creates `status/COMPLETE.txt` and disables its own daily workflow.

Important: FIRMS points are satellite thermal detections, not an official BC Wildfire Service perimeter or official fire-status declaration.

The default expansion rule is a new hotspot more than **500 metres** beyond the prior accumulated hotspot footprint.

## One-time GitHub setup

### 1. Create a free private repository

Create a new GitHub repository, for example `brunswick-fire-animation`, and choose **Private**.

### 2. Add these files

Keep this structure:

```text
fire_animation.py
requirements.txt
.gitignore
.github/workflows/brunswick-fire.yml
data/
frames/
output/
status/
```

### 3. Add your FIRMS key as a secret

In the repository go to:

**Settings → Secrets and variables → Actions → New repository secret**

Name it exactly:

```text
FIRMS_MAP_KEY
```

Paste your NASA FIRMS MAP_KEY as the value. Do not put the key directly in the Python or YAML files.

### 4. Test it once

Open **Actions → Brunswick + Ainslie Wildfire Animation → Run workflow**.

The first run is the longest because it builds the history from July 1. Later runs usually regenerate only the recent tail.

### 5. Find the animation

Open **Releases → Latest Wildfire Animation**.

You will find:

- `Brunswick_Ainslie_SIMPLE.mp4`
- `Brunswick_Ainslie_SLIDESHOW_ONCE.gif`
- `Brunswick_Ainslie_SLIDESHOW.html`
- `Brunswick_Ainslie_LATEST.png`

For the exact behavior of stopping on the last image, use the **HTML slideshow**. It remains on the final frame until you close the screen and has a Replay button.

## Automatic stop

Defaults in `fire_animation.py`:

```python
NO_EXPANSION_DAYS_TO_STOP = 5
EXPANSION_TOLERANCE_METERS = 500
```

Automatic stopping is suppressed if the recent FIRMS refresh fails, so a network/API problem is not mistaken for a fire that stopped expanding.

When the threshold is reached, the workflow publishes the final outputs and disables itself.

To resume later, re-enable the workflow in the Actions tab and delete `status/COMPLETE.txt`.
