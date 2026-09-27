# plan

what's left, in the order i'd do it.

rule for all of it: nothing here is allowed to make a working result worse.
keep the current artifact until a replacement beats it on the same held out
set, and if a regression survives it goes in limitations.md rather than
getting reverted quietly.

| # | what | state |
|---|---|---|
| 1 | landfall land mask | done |
| 2 | INSAT, archive and live | done |
| 3 | say what the numbers actually measure | done |
| ~~4~~ | ~~RI scene features~~ | not viable, measured |
| ~~5~~ | ~~T2 IRRCDO by class balancing~~ | not measurable; fixed another way |
| 6 | before submission | below |

## 1. landfall land mask - done

the model regressed a landfall lat/lon with no coastline in its inputs, so
nothing pulled the answer onto land. 256 km mean error, worse than the 122 km
24 h track error, which is the diagnostic. fixed by taking the point where the
forecast track crosses the coastline, which is what landfall means: 184 km
mean, 114 km median, timing unchanged at 9.0 h.

## 2. INSAT, archive and live - done

### what we got wrong first

we believed an INSAT granule was a latitude strip that moved between campaigns,
and designed around it. the strips are real - they are the rapid-scan sectors
ISRO runs over an active storm - but the full sector is scanned every half
hour, 3DR at :15 and :45, 3D and 3DS at :00 and :30, covering 9.5S to 43.6N.
picking the granule nearest the hour picked a strip every time. that one line
is what made the whole archive usable.

### what exists now

1,207 full-sector scans, INSAT-3D 2014-2016 and 3DR 2017-2025, regridded onto
the GridSat grid at the same 3-hourly slots with the same file naming, so every
model and endpoint reads either sensor unchanged. 505 storm patches paired with
their GridSat twins. `src/ingest/insat_archive.py` builds the archive from a
manifest and survives token expiry, `src/ingest/insat_grid.py` regrids it, and
`src/ingest/insat_live.py` keeps the last 48 h of INSAT-3DS on disk for the
live tab.

### what it changed

| | GridSat-trained | trained on both |
|---|---|---|
| T3 on GridSat | 13.16 kt | 12.96 kt |
| T3 on INSAT | 15.11 kt, bias -4.7 | 13.77 kt, bias -0.7 |
| T1 on GridSat, named F1 | 0.700 | 0.626 |
| T1 on INSAT, named F1 | 0.408 | 0.675 |
| T2 on GridSat | 0.722 | 0.713 |
| T2 on INSAT | 0.601 | 0.612 |

so: T3 trains on both and serves everywhere. T1 keeps two checkpoints, because
the joint detector is much better on INSAT and clearly worse on GridSat, and
the imagery source decides which one answers. T2 stays GridSat-trained because
joint training moved it nowhere on either sensor.

## 3. say what the numbers actually measure - done

T3 had been trained and scored against ADT's own wind estimate while calling it
best track. the whole story is in the engineering log; the short version is that
ADT reports a 1-minute wind and IMD a 3-minute one, 7.6 kt apart, and the model
had learned ADT's scale.

that led to the rest of it. single held-out splits of 14 storms turned out to
move further between draws than the differences we were reading off them, so T2
and T3 are now scored by 5-fold cross-validation grouped by storm, and every
headline number carries a 95% interval from resampling whole storms. that is
also how we learned the T2 CNN had never beaten its physics baseline, and that
averaging the two beats both.

## 4. RI scene features - not viable

the RI model never sees the satellite scene even though a CDO tightening into
an eye is what intensification looks like. it cannot be done on this data:
scene patches are keyed by ADT storm id at ADT analysis times and the forecast
dataset by IBTrACS SID at synoptic hours, and even with a perfect crosswalk the
974 patches would carry about six positive RI cases in test. six cases cannot
move a brier skill score. revisit if the patch archive grows an order of
magnitude.

## 5. T2 IRRCDO - not measurable as planned, improved anyway

the plan was class-balanced sampling scored on macro-F1, and with 7 test
examples we could have trained it but not honestly claimed it improved
anything. cross-validation gave the class 97 examples to be measured on, and
the hybrid took it from 0.33 to 0.53 without anyone targeting it: the
cold-cloud half is better at IRRCDO than the CNN is.

## 6. before submission

- ~~rebuild the deck from `reports/` and export the PDF~~ done. six slides,
  every figure read from a report file, rendered and checked for overflow.
- ~~check every number in the deck against `src/eval/headline_numbers.py`~~ done,
  and it found five stale figures and a bug. the arbiter was itself reporting a
  model that no longer serves; both are fixed and the story is in the log.
- retake the deck screenshot. the dashboard now displays IST and the shot on
  slide 2 still shows UTC, so the caption ("19 May 2020 00:00 UTC") and the
  image have to move together. `Chakravat-deck/shoot.py` points at port 8010.
- a short demo video, since the panel sees a PDF and not a running system
- freeze, then rerun `src/smoke_api.py` and `src/eval/headline_numbers.py` one
  last time

## 7. the last eleven days

picked over attempting microwave imagery, which stays in "not doing" below: six
things that can each be finished and measured, rather than one that might not
land at all.

| | what | why | state |
|---|---|---|---|
| A1 | wind probability swaths | the cone is a centre product; this is an impact one | done |
| B3 | population in the swath over time | how likely, and by when | done |
| A3 | historical analogues | the explanation a forecaster already trusts | done |
| C1 | storm surge | scouted: the gauges to validate it do not exist | dropped |
| - | ERA5 into T2/T3 | measured, no gain, not promoted | done |
| - | six-channel INSAT regrid | 1,207 granules, split window available | done |
| 1 | scene typing on INSAT | 0.61 against 0.72 on GridSat, and INSAT is the live feed | not fixed, diagnosed |
| 2 | depression recall | we miss 6 in 10, and a second operating point costs no retraining | done |
| 3 | prediction interval on T3 | the severe under-read is invisible in a point estimate | done |
| 4 | why the forecast said that | T2 and T3 have Grad-CAM, T4 had nothing | done |
| 5 | SMS-length alert | CAP goes to SACHET and SACHET sends SMS | done |
| 6 | the live loop actually running | the pipeline on a schedule, storm list maintained | |

### 1 is closed as a measured failure

four attempts, none promoted: fitting the cold-cloud half per sensor, adding a
sensor indicator, per-sensor blend weights, and standardising each patch by its
own channel statistics. the last made both sensors worse, which was the clue.
scoring the 505 exactly paired scenes says the gap is the sensor and it lives in
two classes - EMBC 0.69 to 0.47 and IRRCDO 0.58 to 0.39 - while EYE is untouched
at 0.76 against 0.75. a geometric feature transfers between instruments and a
temperature-texture judgement does not.

what shipped instead is honesty at the point of use: `/scene` returns the
measured F1 for the class it just predicted on the sensor it just read, and the
dashboard flags the weak ones. `src/eval_t2_sensors.py` and
`src/eval_t2_stat_variants.py` regenerate every number above.

### 2 and 3 both turned into "say what it costs"

the detector gets a second, more sensitive operating point chosen on validation
by F2, served as `tier=watch`: weak-system recall 0.480 to 0.529 for twice the
false alarms. the finding underneath it is the control run - at threshold 0.05,
eight times the false alarm rate, four in ten weak systems are still missed, so
they are not behind the threshold and no amount of sensitivity will find them.

T3 now serves an interval fitted on out-of-fold residuals and verified like the
cone: 0.500, 0.651 and 0.875 measured against 0.50, 0.67 and 0.90 targets, and
uniform across bands. binned on the prediction rather than the truth, which is
the only honest way to condition it and also the reason the number differs from
the -12.7 kt in limitations.

## storm surge: scouted and dropped

surge is what kills in the bay of bengal, so it was worth asking whether we
could forecast it honestly. the question is not whether a surge model can be
written - it is whether the observations exist to check one. surge is validated
against tide gauge residuals, observed sea level minus predicted tide, and
`src/scout_tide_gauges.py` counts how many of our landfalls have a gauge near
enough and recording at the time.

the answer is no, and it is not close. UHSLC lists **four** gauges for the whole
of india: minicoy, cochin, port blair and vishakhapatnam. exactly one of those
is on the east coast, which is where 1999 odisha, phailin, fani and amphan came
ashore. across the entire archive there are 19 severe landfalls near a working
gauge and 12 of them are bangladesh; india contributes 2.

so a surge model here would be fitted and checked almost entirely on bangladeshi
landfalls, on fourteen storms, for an indian problem statement. that number
would have been the only unverified figure in the project. not built, and the
scouting report is kept so the decision can be checked rather than taken on
trust.

## not doing

- **more GridSat years.** 4x the data moved T1 F1 from 0.461 to 0.518. sample
  count isn't the constraint.
- **chasing the 24 h track target.** 126 km against a 120 km goal. closing 6 km
  with more capacity on 183 training storms is overfitting.
- **beating IMD.** not the claim.
- **microwave imagery.** 89 GHz sees the eyewall through the cirrus that hides
  it from infrared, and it is the obvious next sensor for intensity. it is also
  a new dataset, a new geometry and a new set of failure modes, and there are
  twelve days. it is the first thing i would do next.

## what this leaves

the honest closing position is that the constraint is not modelling capacity.
it is how much of this basin has been observed at all: about five named storms a
year, 943 labelled scenes, 51 held-out storms. that is why the last week went
into measuring what we have properly - cross-validation, intervals, a benchmark
against ADT, the same storms through two sensors - rather than into another
model.
