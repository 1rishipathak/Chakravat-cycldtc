# chakravat

cyclone detection, classification and forecasting for the north indian ocean.
built for SIH problem statement 26070 (ministry of earth sciences / IMD).

the problem statement asks for identification, classification and prediction
from multi-source satellite data. i split that into four tasks:

- T1, find storms in a satellite image and pin the centre
- T2, classify the dvorak cloud pattern
- T3, estimate intensity from the image alone
- T4, forecast track and intensity out to 72 h

everything is scored on held out seasons 2020-2025, 51 storms that no model
here has seen.

not an IMD product. official warnings come from RSMC new delhi.

## running it

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install numpy pandas scikit-learn matplotlib joblib xarray netCDF4 h5py timm fastapi uvicorn shapely requests
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

model weights are in the v1.0 release (they're 111 MB each so github won't take
them in the repo). unzip into the repo root so they land in `artifacts/`, then

```bash
.venv/Scripts/python.exe -m uvicorn api.main:app --port 8000
```

and open http://localhost:8000

if you want to retrain instead of downloading, the `src/train_*.py` scripts do
that, but you'll need the ~9 GB of source data first via `src/ingest/`.

## building a different frontend

`fixtures/` is the contract. it has a real captured response from every
endpoint, 15 files and 70 KB total, so the whole UI can be built with no
backend, no dataset and no GPU. the filenames say which endpoint each came
from:

| fixture | endpoint |
|---|---|
| `seasons.json`, `storms_2020.json` | `/api/seasons`, `/api/storms?season=` |
| `storm_amphan.json` | `/api/storm/{sid}` |
| `forecast_amphan_05/067/09.json` | `/api/storm/{sid}/forecast?level=`, all three levels |
| `landfall_lands.json`, `landfall_none.json` | `/api/storm/{sid}/landfall`, both branches |
| `bulletin_amphan.txt` | `/api/storm/{sid}/bulletin`, plain text not json |
| `scene_amphan.json`, `intensity_amphan.json` | T2 and T3 on one patch |
| `pipeline.json` | `/api/vision/pipeline`, the whole chain |
| `vision_status.json`, `scene_meta.json`, `skill.json` | status, scene bounds, all metrics |

`web/index.html` is a reference implementation, not a constraint. one file, no
build step. regenerate the fixtures against a running server with

```bash
.venv/Scripts/python.exe src/dump_fixtures.py
```

CORS is already open, so a dev server on another port works without a proxy.

## results

| task | ours | baseline it beats |
|---|---|---|
| T1 detection, all systems | F1 0.586, 74 km fix | coldest pixel: F1 0.123, 98 km |
| T1 detection, 34 kt and up | F1 0.700, recall 0.759, 71 km | - |
| T2 dvorak scene | macro-F1 0.689, acc 0.764 | cold cloud stats: 0.563 |
| T3 intensity from image | RMSE 10.95 kt, MAE 7.36 | cold cloud stats: 21.06 kt |
| T4 track + intensity @ 24 h | 121.9 km / 6.92 kt | CLIPER: 142.1 km / 9.04 kt |

T1 gets reported twice on purpose. the pooled number includes depressions that
often have no organised signature in infrared at all. the 34 kt+ number is the
storms IMD actually names and warns on. neither one alone is the honest answer.

### forecast skill vs CLIPER

CLIPER is climatology plus persistence, it's the benchmark operational centres
score skill against. beating it is the thing that matters, not beating zero.

| lead | track | ours | skill | intensity | ours | skill |
|---|---|---|---|---|---|---|
| 6 h | 35.6 km | 33.6 | +5.7% | 3.06 kt | 2.78 | +9.2% |
| 12 h | 69.1 km | 60.2 | +12.8% | 5.37 kt | 4.50 | +16.3% |
| 24 h | 142.1 km | 121.9 | +14.2% | 9.04 kt | 6.92 | +23.5% |
| 48 h | 285.6 km | 254.0 | +11.1% | 13.96 kt | 11.89 | +14.9% |
| 72 h | 405.2 km | 379.1 | +6.4% | 16.42 kt | 15.01 | +8.6% |

skill is positive everywhere and peaks at 24 h, which is the lead time an
evacuation call actually gets made on.

### cone coverage

a 67% cone should contain about 67% of the true positions. ours at 24 h:

| lead | 50% | 67% | 90% |
|---|---|---|---|
| 6 h | 45% | 63% | 87% |
| 24 h | 46% | 65% | 86% |
| 72 h | 56% | 73% | 95% |

slightly tight at short leads, slightly generous at long ones. reported either
way, because a cone whose label doesn't mean anything is worse than no cone.

### landfall

| | ours | baseline |
|---|---|---|
| timing | 9.04 h | 14.88 h |
| position | 184 km mean, 114 km median | 286 km |
| intensity at coast | 11.77 kt | 13.49 kt |
| will it land in 72 h | POD 0.80, FAR 0.20, CSI 0.65 | - |

position earlier used to be 256 km and was the worst thing in the project. the model
regressed a lat/lon with no coastline anywhere in its inputs, so nothing pulled
the answer onto land. only 28% of predictions landed within 25 km of a coast.
fixed by taking the point where the forecast track crosses the coastline
instead, which is what landfall actually means.

### rapid intensification

brier skill score +0.096 over climatology. positive so it's real, but thin. RI
is one of the hardest open problems in the field and i'm not going to oversell
a 10% improvement on the base rate.

## data

| source | what it gives | access |
|---|---|---|
| GridSat-B1 (NOAA) | global IR + water vapour, 8 km, 3-hourly | no auth |
| CIMSS ADT archive | dvorak scene labels, 8212 of them | no auth |
| IBTrACS | best track, our ground truth | no auth |
| ERA5 (copernicus) | shear, humidity, SST, steering flow | free account |
| digital typhoon | pretraining frames for T3 | kaggle account |
| INSAT-3DR (MOSDAC) | 4 km imagery | approved account |

MOSDAC approval came through late and it turned out we never needed it. GridSat
and the ADT archive cover T1 and T2 with no authentication at all, so every
number above was produced without INSAT. `src/verify_insat.py` shows INSAT
imagery running through the trained models unretrained, which is the point:
switching sensors is a data loader change, not a rebuild.

see attribution.md for licences. ERA5 has one that has to be reproduced word
for word.

## what doesn't work

read limitations.md, it's the honest list. short version:

- detection misses about 6 in 10 depressions
- T3 still reads severe storms ~12 kt low in the 64-89 kt band
- RI is barely skilful
- IRRCDO scene class is basically not learned, 7 test examples
- test sets are small because the basin only makes ~5 storms a year

## a few things worth knowing

**splits are by storm, never by row.** one cyclone gives a fix every 6 hours and
consecutive fixes are near identical, so a random row split puts the same storm
on both sides and the model just memorises. we measured the gap: random splits
inflate 24 h intensity by about 10%.

**every model gets a physics baseline, not just a null one.** cold cloud
statistics on the pixel histogram is a genuinely hard opponent, and that's the
point. it caught a real bug once, T3 was scoring 17.08 kt while the baseline got
15.73, which is how we found out the model had collapsed to predicting the mean.

**the IBTrACS "north indian" file isn't all north indian.** 25 of 291 storms are
western pacific typhoons that wandered across 100E. filtering them out took us
to 266 storms and moved RI skill from +0.055 to +0.096.

## layout

```
src/ingest/     downloading and parsing each data source
src/features/   feature engineering, ERA5 environment
src/models/     CLIPER, residual booster, ensemble, RI, landfall, coastline
src/vision/     patch extraction, detection dataset
src/train_*.py  one script per model
src/pipeline.py imagery in, forecasts out, the whole chain
api/            fastapi server
web/            dashboard
reports/        every verified number, as json
```
