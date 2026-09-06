# plan

what's left, in the order i'd do it.

rule for all of it: nothing here is allowed to make a working result worse.
keep the current artifact until a replacement beats it on the same held out
set, and if a regression survives it goes in limitations.md rather than
getting reverted quietly.

| # | what | why now | effort |
|---|---|---|---|
| 1 | landfall land mask | done, see below | - |
| 2 | INSAT-3D fine tune | MOSDAC unblocked it | 1-2 days |
| ~~3~~ | ~~RI scene features~~ | not viable, measured | - |
| ~~4~~ | ~~T2 IRRCDO~~ | not measurable, measured | - |

## 1. landfall land mask - done

the model regressed a landfall lat/lon with no coastline in its inputs, so
nothing pulled the answer onto land. 256 km mean error, worse than the 122 km
24 h track error, which is the diagnostic.

three options were on the table:

- snap the prediction to the nearest coast point. cheap, fixes "not on land"
  but not "wrong bit of coast", which is most of the error.
- reparameterise the target as distance along the coastline. principled but
  needs a full retrain.
- intersect the forecast track with the coastline. landfall *is* where the
  track meets land, so this inherits the track forecast's 122 km error instead
  of accumulating a separate one, and it removes a model rather than adding
  one.

went with the third, with snapping as fallback when the track never reaches
land inside 72 h.

results: 256 km to 184 km mean, 211 km to 114 km median. timing stayed at
9.04 h so nothing regressed. coastline is natural earth 1:10m clipped to the
basin and vendored at `data/static/coastline_nio.geojson`, 550 KB, so the demo
never needs a network for it.

## 2. INSAT-3D fine tune

### what we already proved

`src/ingest/insat.py` authenticates against MOSDAC and pulls granules. one auth
attempt only, no retries, because three consecutive failures locks the account
for an hour.

`src/ingest/insat_regrid.py` puts a granule on the GridSat grid so downstream
code can't tell the difference. `src/verify_insat.py` runs the existing models
on INSAT imagery with no retraining and gets sane numbers on amphan.

channels map straight across, TIR1 is our IR and WV is our WV, so the three
channel input carries over unchanged. this is a data problem not a modelling
one.

### two traps in the data

**count 1023 is a fill value and the lookup table is inverted.** LUT[0] is
340.1 K and LUT[1023] is 179.9 K, so no-data converts to the *coldest*
temperature. about 70% of a granule is fill. left alone the detector would see
a basin sized sheet of deep convection and find storms everywhere, with a
perfectly healthy loss curve. found this by checking a known clear sky point
rather than trusting the sector mean.

**a granule is a strip, not the whole sector.** typically 17-19 degrees of
latitude, and ISRO moves it between campaigns. we measured 2.2-19.7N one
morning in may 2020 and 7.7-26.1N that same afternoon. the catalogue doesn't
expose it so you have to read it off the array.

### budget

| | named storms only | everything |
|---|---|---|
| storm days | 189 | 477 |
| granules at ~5 usable slots/day | ~945 | ~2385 |
| download | ~6 GB | ~15 GB |
| transfer at 6.6 MB/s | 15-25 min | 40-60 min |

MOSDAC caps at 5000 files/day/user so either fits. add ~40 min to rebuild the
patch cache and 3-6 h to fine tune the three models. call it a working day.

### the actual risk

T2's training data halves. only 483 of 974 patches are in the INSAT era and
every class roughly halves, CRVBND 377 to 185, EYE 141 to 77, IRRCDO 97 to 54.
T2 is already sensitive to dataset changes (that's the 0.723 to 0.689 episode)
so fine tuning on half the data at higher resolution is not obviously a win.

T3's expected gain is low because we already proved resolution wasn't its
binding constraint, calibration was.

and 10-20% of fixes fall outside whatever strip that year's campaign used.

### rule

keep the GridSat checkpoints. only serve an INSAT model where it wins on a
common held out set. if it wins on some tasks and loses on others, serve per
task winners and say so.

### the part that actually matters for a demo

INSAT-3DR is a live feed. right now the honest answer to "could this run
tomorrow" is no, because GridSat is a delayed archive. one recent scene through
the pipeline changes that answer. worth more to a panel than a decimal place.

## 3. RI scene features - not viable

idea was fine. the RI model only sees best track and ERA5, never the satellite
scene, even though a CDO tightening into an eye is what intensification looks
like. adding the T2 scene probabilities should help.

it can't be done on this data, for two reasons i measured instead of guessing.

**no crosswalk between the datasets.** scene patches are keyed by ADT storm id
(`200501B`) at ADT analysis times like 01:00 and 11:30. the forecast dataset
uses IBTrACS SIDs at synoptic hours. exact key overlap is zero rows. `vmax_kt`
in patches.csv is ADT's own estimate, not a best track join, so there's nothing
to reuse.

**even a perfect crosswalk leaves nothing to learn from:**

| | |
|---|---|
| RI positives in the whole dataset | 162 (105 train / 28 test) |
| GridSat patches, hard ceiling on rows that could carry the feature | 974 = 18% |
| RI positives among those at the 3.79% base rate | ~36 |
| split train/test | ~23 train, ~6 test |

six positive test cases can't move a brier skill score in any meaningful way.
not attempted. revisit if the patch archive ever grows an order of magnitude.

## 4. T2 IRRCDO - not measurable

F1 0.333, rarest and most ambiguous class. plan was class balanced sampling
scored on macro-F1.

the counts kill the verification, not the training:

| split | n | IRRCDO |
|---|---|---|
| train | 718 | 82 |
| test | 123 | 7 |

with 7 test examples, moving F1 from 0.333 to 0.5 means getting one more patch
right. we could train it, we couldn't honestly claim it improved anything. not
attempted.

## not doing

- **more GridSat years.** 4x the data moved T1 F1 from 0.461 to 0.518. sample
  count isn't the constraint.
- **chasing the 24 h track target.** 121.9 km against a 120 km goal, down from
  141 before ERA5. closing 2 km with more capacity on 183 training storms is
  overfitting.
- **beating IMD.** not the claim.

## what this leaves

INSAT is the only remaining work with real headroom, and both of the dead items
above point the same way: the constraint across this project is data volume now,
not modelling. that's a better closing argument than two unmeasurable
experiments would have been.
