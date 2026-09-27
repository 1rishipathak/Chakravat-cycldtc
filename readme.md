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
endpoint, 23 files and 142 KB total, so the whole UI can be built with no
backend, no dataset and no GPU. the filenames say which endpoint each came
from:

| fixture | endpoint |
|---|---|
| `seasons.json`, `storms_2020.json` | `/api/seasons`, `/api/storms?season=` |
| `storm_amphan.json` | `/api/storm/{sid}` |
| `forecast_amphan_05/067/09.json` | `/api/storm/{sid}/forecast?level=`, all three levels |
| `landfall_lands.json`, `landfall_none.json` | `/api/storm/{sid}/landfall`, both branches |
| `bulletin_amphan.txt`, `bulletin_amphan_hi.txt` | `/api/storm/{sid}/bulletin?lang=`, plain text not json |
| `exposure_amphan.json` | `/api/storm/{sid}/exposure`, places in the threat zone |
| `cap_amphan.xml` | `/api/storm/{sid}/cap.xml`, a CAP 1.2 alert |
| `live_status.json` | `/api/live/status`, what the live INSAT feed holds |
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

| task | ours | what it is measured against |
|---|---|---|
| T1 detection, all systems | F1 0.586, 74 km fix | coldest pixel: F1 0.123, 98 km |
| T1 detection, 34 kt and up | F1 0.700, recall 0.759, 71 km | - |
| T2 dvorak scene | macro-F1 0.710 (0.672-0.741) | cold cloud stats: 0.650, CNN alone: 0.664 |
| T3 intensity from image | RMSE 12.96 kt (11.5-14.4) | ADT on the same scenes: 13.95 kt |
| T4 track + intensity @ 24 h | 126 km / 7.3 kt | CLIPER: 142 km / 9.0 kt |

two things about that table.

**the intervals are the point.** T2 and T3 are scored by 5-fold cross-validation
grouped by storm - every patch predicted by a model that never saw its storm -
and every number carries a 95% interval from resampling whole storms. a single
held-out split of 14 storms moves further between draws than most of the
differences anyone argues about, which we learned by believing one: it had said
T2 beat its baseline 0.689 to 0.563, and the truth is 0.664 to 0.650, a tie.
what serves now is the average of the two, which beats either.

**T3's truth is IMD best track**, interpolated to the scene time. it used to be
ADT's own estimate, which is a 1-minute wind sitting 7.6 kt above IMD's
3-minute one. ADT is now the benchmark rather than the target, and on it the
two are level while ours is the unbiased one.

T1 gets reported twice on purpose. the pooled number includes depressions that
often have no organised signature in infrared at all. the 34 kt+ number is the
storms IMD actually names and warns on. neither one alone is the honest answer.

the forecast numbers are the ones the API serves, not the best row in the
selection table, and they carry intervals too: +11.2% on track at 24 h
(+8.3 to +14.4) and +19.5% on intensity (+10.0 to +27.3). at 72 h the interval
crosses zero, so we don't claim skill that far out.

### forecast skill vs CLIPER

CLIPER is climatology plus persistence, it's the benchmark operational centres
score skill against. beating it is the thing that matters, not beating zero.

| lead | CLIPER | ours | skill | 95% interval |
|---|---|---|---|---|
| 6 h | 35.6 km | 33.5 | +5.6% | +4.1 to +7.2% |
| 12 h | 69.1 km | 62.2 | +9.9% | +7.8 to +12.4% |
| 24 h | 142.1 km | 126.2 | +11.2% | +8.3 to +14.4% |
| 48 h | 285.6 km | 259.2 | +9.2% | +4.9 to +13.8% |
| 72 h | 406.5 km | 388.3 | +4.5% | -1.4 to +11.3% |

| lead | CLIPER | ours | skill | 95% interval |
|---|---|---|---|---|
| 6 h | 3.05 kt | 2.71 | +11.4% | +5.0 to +16.5% |
| 12 h | 5.37 kt | 4.31 | +19.8% | +12.9 to +25.2% |
| 24 h | 9.03 kt | 7.26 | +19.5% | +10.0 to +27.3% |
| 48 h | 13.92 kt | 11.90 | +14.5% | +5.0 to +22.4% |
| 72 h | 16.41 kt | 15.22 | +7.3% | -1.4 to +15.4% |

skill peaks at 24 h, which is the lead time an evacuation call actually gets
made on, and the interval stays clear of zero through 48 h. at 72 h it doesn't,
so that row is a number we have rather than a claim we make.

these are the forecasts the API serves - the ensemble mean. the selection table
in `reports/final_results.json` has a blend that scores 121.9 km at 24 h, but
nothing calls it, and quoting a number nobody can reproduce through the API
would be quoting the wrong thing.

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

position used to be 256 km and was the worst thing in the project. the model
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
| IBTrACS | best track, our ground truth and the wind radii | no auth |
| ERA5 (copernicus) | shear, humidity, SST, steering flow | free account |
| digital typhoon | pretraining frames for T3 | kaggle account |
| INSAT-3D and 3DR (MOSDAC) | 1,207 full-sector scans, 2014-2025, on our grid | approved account |
| INSAT-3DS (MOSDAC) | the live feed, a scan every 3 h about an hour behind | approved account |
| natural earth | coastline, india point-of-view boundaries, populated places | no auth |

the order matters. everything was built on GridSat first, because it needs no
approval and a system that depends on one you don't control isn't a system.
MOSDAC access then came through, and `src/ingest/insat_archive.py` mirrors the
INSAT archive onto the same grid at the same 3-hourly slots, so the two sensors
can be compared storm for storm. T2 and T3 now train on both; detection is
still being decided.

the one thing to know about the INSAT archive: only two granules an hour are
the full sector. the rest are rapid-scan strips ISRO runs over an active storm,
and picking the granule nearest the hour gets you a strip every time. 3DR scans
the sector at :15 and :45, 3D and 3DS at :00 and :30.

see attribution.md for licences. ERA5 has one that has to be reproduced word
for word.

## what doesn't work

read limitations.md, it's the honest list. short version:

- detection misses about 6 in 10 depressions
- intensity still reads the strongest storms low: -8 kt at 64-90, -13 at 90+
- scene typing is much weaker on INSAT imagery (0.61) than on GridSat (0.72),
  and the live feed is INSAT
- RI is barely skilful
- IRRCDO is the weakest scene class, F1 0.53
- 72 h forecast skill is positive but its interval crosses zero
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
src/ingest/     downloading and parsing each data source, INSAT archive and live
src/features/   feature engineering, ERA5 environment
src/models/     CLIPER, residual booster, ensemble, RI, landfall, coastline, impact
src/vision/     patch extraction, detection dataset, best-track crosswalk
src/train_*.py  one script per model
src/cv_*.py     cross-validation by storm: model choices are made here
src/eval/       metrics, splits, confidence intervals, headline_numbers.py
src/pipeline.py imagery in, forecasts out, the whole chain
src/smoke_api.py   hits every endpoint and checks the shape of the answer
api/            fastapi server, alerts and delivery
web/            dashboard and live alert page
reports/        every verified number, as json
```
