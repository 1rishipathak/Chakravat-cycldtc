# limitations

everything here was measured on storms the model never saw. these are all
problems we found ourselves rather than ones somebody pointed out.

knowing when a system is wrong is more useful than a higher headline number,
so this list exists on purpose.

## 1. detection misses most weak systems

F1 0.586, recall 0.540 pooled over everything. coldest pixel baseline is 0.123.

the pooled number is dominated by systems nobody warns about. 79% of test fixes
are depressions or deep depressions under 34 kt, and a 20 kt depression often
has no organised convective signature in infrared, it looks like ordinary
monsoon convection.

recall by band:

| band | n | recall |
|---|---|---|
| depression, under 28 kt | 218 | 0.394 |
| deep depression, 28-33 | 88 | 0.693 |
| cyclonic storm, 34-47 | 49 | 0.776 |
| severe CS and up, 48+ | 34 | 0.735 |

restricted to the 83 fixes at 34 kt and above, F1 0.700, recall 0.759, median
fix 71 km. both numbers get printed on every run. quoting only the second would
be dishonest and quoting only the first understates it.

we still miss about 6 in 10 depressions. that's partly deliberate from the
intensity weighting below, and it's still a real miss rate.

### is that a tuning choice? no, and we checked

the served threshold is 0.25, picked on validation for F1, which is the right
objective when a detection becomes a track and a track becomes a forecast. for
asking "is anything forming out there" a miss costs more than a false alarm, so
the same checkpoint gets a second operating point picked on validation for F2.
nothing is retrained and the served point does not move.

| tier | threshold | precision | weak recall | false alarms per scene | named F1 |
|---|---|---|---|---|---|
| warn, served | 0.25 | 0.640 | 0.480 | 0.33 | 0.700 |
| watch | 0.15 | 0.495 | 0.529 | 0.66 | 0.647 |

so the watch tier doubles the false alarms to move weak-system recall five
points. that is a fair trade for a screening view and a bad one for a forecast,
which is why both exist and the tier is explicit in the API.

the number that matters more is the third row we ran, the most permissive
threshold on the grid:

| ceiling | 0.05 | 0.204 | 0.595 | 2.73 | - |

eight times the false alarm rate of the served point, and four in ten weak
systems are **still** missed. they are not systems the network nearly saw and
narrowly rejected. they are systems with no signature it can find, which is what
"a 20 kt depression looks like ordinary monsoon convection" means in practice.
more sensitivity is not the lever; a different sensor would be.

`reports/t1_operating_points.json` holds all of it,
`src/t1_operating_points.py` regenerates it.

what we tried: extending the archive from 4 seasons to 14 (224 to 921 scenes)
moved F1 only 0.461 to 0.518. poor return on 4x the data, which was good
evidence the limit wasn't sample count.

what worked was changing what the loss cares about. each storm now gets weighted
by intensity, `w = clip((vmax/34)^1.5, 0.3, 3.0)`. F1 went 0.518 to 0.586,
recall 0.437 to 0.540, and every band improved including depressions, even
though depressions got weighted down to 0.3x.

the 0.90 target isn't reachable on 8 km imagery and more of the same data won't
get there.

## 2. intensity reads the strongest storms low

worst limitation operationally, because under-reading a severe cyclone is the
wrong direction to be wrong in.

truth here is IMD's best track, interpolated to the scene time. it used to be
ADT's own estimate, which is a 1-minute wind and sits 7.6 kt above IMD's
3-minute one; that mistake and its fix are the first entry in the engineering
log. every number below is from 5-fold cross-validation grouped by storm, so all
943 patches are scored by a model that never saw their storm, and the intervals
come from resampling whole storms 2,000 times.

| | RMSE | 95% interval | bias | IMD category exact |
|---|---|---|---|---|
| predict the mean | 24.86 kt | 21.6 to 28.3 | +0.8 | 13% |
| cold-cloud statistics | 19.37 kt | 17.2 to 21.7 | +0.5 | 29% |
| ADT, on the scenes it covers | 13.95 kt | 12.5 to 15.3 | +7.1 | 40% |
| chakravat | 12.96 kt | 11.5 to 14.4 | -0.1 | 39% |

against ADT on its own 719 scenes the difference is -2.6 to +1.6 kt, so the two
are level and ours is the unbiased one. that is the claim: comparable to the
operational objective method, not better than it. ADT still gets the category
right slightly more often (40% against 39%), which is worth saying out loud
because our RMSE is the better number and the category is the one a bulletin
prints.

the model serving this is trained on GridSat and INSAT together. that costs
nothing here - 13.16 kt for GridSat-only training against 12.96 for joint on the
same patches - and it is what makes the same model usable on the live INSAT
feed, where a GridSat-only model reads 15.11 kt with a -4.7 kt bias against
13.77 and -0.7 for this one.

the band table is the part that matters:

| truth band | <34 | 34-48 | 48-64 | 64-90 | 90+ |
|---|---|---|---|---|---|
| patches | 336 | 273 | 105 | 133 | 96 |
| bias (kt) | +5.6 | +1.7 | -1.6 | -7.8 | -12.7 |

the overall bias of -0.1 kt hides all of it. weak systems come out too strong and
the strongest too weak, and the two cancel. the joint model is slightly worse at
the top end than the GridSat-only one it replaced (-12.7 against -11.3) and
better at the bottom; we took the trade for the lower RMSE and the working live
feed, and the worst single case we have seen is Amphan at its peak, read 29 kt
low from one scene.

### so the estimate now comes with an interval

a single number with a 13 kt error is not an answer. the estimate is served with
a band, fitted on the out-of-fold residuals we already had - every patch
predicted by a model that never saw its storm - and verified the same way the
forecast cone is:

| level | target | measured coverage | mean width |
|---|---|---|---|
| 50% | 0.50 | 0.500 | 15.8 kt |
| 67% | 0.67 | 0.651 | 23.5 kt |
| 90% | 0.90 | 0.875 | 40.5 kt |

slightly tight at the top two levels, and uniform across bands: at 90% the
coverage runs 0.854 to 0.882 whichever band the prediction lands in, so it is
calibrated conditionally and not just on average.

**the bands are on the prediction, not on the truth, and that distinction is the
whole point.** the -12.7 kt figure above is conditioned on truth: when a storm
really is 90 kt or more, we read it low. that is not usable at inference, where
nobody knows the truth. conditioned on what the model *said*, the residual
median in the top band is +2.1 kt and the 67% offsets are -12.8 and +19.7 - a
band that leans upward, which is the same under-read seen from the other end.
both numbers are correct and they are answers to different questions; binning by
truth would have produced an interval that looks far better than it is.

the case it misses is the one we already name: Amphan at peak, read 96 kt
against 125, outside even the 90% band. that is one of the 12.5% the level does
not claim, and it is the most important storm in the set, which is worth saying
out loud rather than leaving for someone to find.

`reports/t3_intervals.json`, regenerated by `src/t3_intervals.py`.

### the calibration that fixes the bands costs accuracy

that pattern is regression toward the mean: fit predicted against truth and the
slope comes out near 0.75, so inverting the line should undo it. it does, for
the bands - 90+ goes from -13 kt to +2 - and it costs 2.4 kt of RMSE, 13.0 to
15.3, which holds even when the line is fitted on four folds of held-out
predictions rather than one small validation set.

so it is not served. the error we can measure is worse with it, and the bias we
can measure is worse without it; we chose the one the primary metric picks and
printed the band table rather than hiding either.

an earlier version of this file argued the residual error wasn't a modelling
problem at all: eye diameter is at or below the 8 km pixel scale, so a model
can't read detail the sensor never recorded. that was wrong, and testing it is
what showed the mapping was the problem rather than the resolution - within the
64 kt+ band the model correlates +0.883 with truth and reproduces its spread.

## 3. rapid intensification is barely skilful

brier skill score +0.096 against climatology. positive so it's real, but at the
operating point it catches well under half of RI events and about 4 in 5 alarms
don't verify.

base rate in this basin is 3.27%. this is "slightly better than quoting the
climatological rate", not a working RI predictor.

## 4. landfall position, mostly fixed now

timing is good, 9.04 h mean error vs 14.88 h for distance over speed.

position used to be the failure. the model regressed a lat/lon with no
coastline anywhere in its inputs so nothing constrained the answer to be on
land. 256 km mean error, worse than our 122 km 24 h track error, and that
inversion was the tell.

we measured the defect before fixing it: only 28% of predicted landfall
coordinates fell within 25 km of any coastline.

fixed by taking the point where the forecast track crosses the coast, which is
what landfall means. that inherits the track forecast's error instead of
accumulating a separate one.

| method | mean | median |
|---|---|---|
| raw regression | 256 km | 211 km |
| snapped to nearest coast | 234 km | 184 km |
| track x coastline (229 of 303 fixes) | 160 km | - |
| hybrid, what ships | 184 km | 114 km |

still not good enough to evacuate a specific village on. timing and the
probability are the parts that carry weight.

## 5. the dvorak labels are algorithm output, and it shows

scene labels come from the CIMSS ADT archive, 8,212 labelled north indian scenes
2003-2025. a model trained on them learns to reproduce ADT, not a human
forecaster. we think that's the right target, since the problem statement asks
for objective automation and ADT is the operational objective standard, but it
is a different claim from "matches an expert" and we don't make the second one.

it has a sharper consequence than we expected. ADT assigns scene type with rules
on cloud-top temperature, so a logistic model on temperature statistics
reproduces the labels about as well as a CNN does. cross-validated over all 974
patches:

| | macro-F1 | 95% interval | accuracy |
|---|---|---|---|
| cold-cloud statistics | 0.650 | 0.604 to 0.691 | 0.659 |
| CNN alone | 0.664 | 0.627 to 0.697 | 0.686 |
| the two averaged, what serves | 0.710 | 0.672 to 0.741 | 0.721 |

the CNN does not beat the baseline. what it does is fail on different scenes -
far better on EYE, worse on SHEAR and IRRCDO - so averaging their probabilities
beats either alone by 0.029 to 0.065 macro-F1. equal weights, nothing tuned on
the held-out folds.

per class, the hybrid: SHEAR 0.82, EYE 0.78, CRVBND 0.73, EMBC 0.69,
IRRCDO 0.53. IRRCDO is the rarest and most visually ambiguous class, 97 patches
in the whole archive, and it is still the weakest thing in T2.

an earlier single-split run said 0.689 for the CNN against 0.563 for the
baseline. it isn't comparable and it was flattering: one split of 123 patches
moves a long way between draws, which is why everything here is cross-validated
now.

## 6. the pipeline inherits its own upstream errors

run from imagery alone on amphan, tauktae and biparjoy, the chain finds the
centre within 39-66 km and puts 11 of 12 forecast positions inside their stated
cone. intensity from the image came within 5-6 kt on amphan and tauktae and
19 kt low on biparjoy.

but the forecast is driven by the detected track, not best track, so detection
scatter propagates into the motion estimate, and the one position that fell
outside its cone was the 6 h step on biparjoy, where the cone is only 29 km
wide and the detection was 39 km off.

two features have no imagery equivalent, central pressure and distance to land.
they get passed as missing rather than guessed. the boosted trees handle NaN
natively. guessing would have hidden the gap.

## 7. the two sensors do not behave the same

GridSat is 8 km and 3-hourly; INSAT-3D, 3DR and 3DS are 4 km and half-hourly,
and INSAT is what a live system in india would actually read. both are on the
same grid here, at the same slots, for the same storms, which is what lets the
difference be measured rather than argued about.

it is not small. a model trained only on GridSat, applied to INSAT imagery:

| | on GridSat | on INSAT |
|---|---|---|
| T3 intensity, RMSE | 13.50 kt | 15.11 kt, bias -4.7 |
| T1 detection, F1 on named storms | 0.700 | 0.408 |
| T2 scene, macro-F1 | 0.722 | 0.601 |

training on both fixes T3 (13.77 kt on INSAT, bias -0.7, and no cost on
GridSat) and fixes detection (0.675 against 0.408 on INSAT), so both of those
now train on both sensors. detection keeps two checkpoints rather than one,
because the joint detector loses 0.07 F1 on GridSat and nothing that works is
allowed to get worse: GridSat scenes get the GridSat detector, INSAT and live
scenes get the joint one.

T2 is the one we could not fix, and we now know exactly what it is.

every one of the 505 INSAT patches has a GridSat twin: same storm, same time,
same ADT label, the same scene through two instruments. scoring on those pairs
removes scene composition from the comparison and leaves only the sensor, and
the gap survives it - 0.725 on GridSat against 0.601 on INSAT. the two sensors
give the same answer on the same scene only 67% of the time, and where they
disagree GridSat is right two and a half times more often.

the per-class table says what is actually broken:

| class | on GridSat | on INSAT | what the class is |
|---|---|---|---|
| EYE | 0.76 | 0.75 | a hole in the cloud |
| SHEAR | 0.84 | 0.76 | a displaced centre |
| CRVBND | 0.75 | 0.64 | a band with a shape |
| EMBC | 0.69 | 0.47 | how cold the overcast is |
| IRRCDO | 0.58 | 0.39 | how uniform it is |

an eye is a geometric feature and it transfers intact. "is this central dense
overcast embedded or irregular" is a judgement about brightness temperature and
texture, and it does not transfer at all. that is not a bug we can normalise
away, and we tried: standardising each patch by its own channel statistics
removes the offset between the sensors, and it made both sensors worse (0.722
to 0.703 on GridSat, 0.601 to 0.560 on INSAT) because it also removes the
absolute temperature those two classes are defined by. fitting the cold-cloud
half separately per sensor lifts that half a long way on INSAT, 0.415 to 0.505,
and moves the blend almost nowhere, 0.601 to 0.608. a per-sensor blend weight
was worse than the fixed one. training the CNN on both sensors gets 0.612.

### a fifth attempt, with a channel we had all along

every INSAT granule carries six channels and the first regrid kept two. the 12.0
micron window was sitting on disk unused, and TIR1 minus TIR2 - the split window
- measures precisely the property the two failing classes are defined by: how
optically thick the cloud top is. thin cirrus lets the 12 micron channel read
warmer, opaque convection looks the same to both.

it separates them, exactly as the physics says it should. averaged over the 505
INSAT patches, in the storm core:

| scene | split window in the core |
|---|---|
| EYE | 1.10 K |
| EMBC | 1.32 K |
| CRVBND | 2.36 K |
| IRRCDO | 2.58 K |
| SHEAR | 2.60 K |

an embedded centre sits under an opaque top and an irregular CDO is ragged and
semi-transparent, and the channel sees the difference where a single window
channel cannot. the whole-patch mean separates almost nothing (2.55 to 2.86); it
is the core that carries it.

feeding those statistics to the INSAT half of the hybrid moves it 0.608 to
0.623, with GridSat bit-identical because GridSat has no second window channel
and its model is literally unchanged. EMBC goes 0.48 to 0.51 and IRRCDO 0.40 to
0.41.

that gain sits inside the interval of the thing it beats, [0.550, 0.655], so by
our own rule it is a measured direction and not a result. it is also not in the
serving path: putting it there means carrying a second channel to inference and
a new failure mode on the live feed, which is a real cost for 0.015 we cannot
demonstrate. the finding worth keeping is the diagnosis - the gap is a cirrus
discrimination problem, the instrument to fix it exists, and a properly
sample-supported model of it is the next thing to build rather than a fifth
variation on this one. `src/eval_t2_split_window.py` regenerates all of it.

four attempts and a fifth that points somewhere, so the GridSat-trained hybrid
still serves everywhere and we say what it costs instead. the scene endpoint now returns the measured F1 for
the class it just predicted on the sensor it just read, so a live INSAT reading
of EMBC arrives carrying "0.47 here against 0.69 on GridSat" rather than
arriving bare. `reports/t2_sensor_paired.json` has the table;
`src/eval_t2_sensors.py` regenerates it.

## 7b. microwave: the instrument that should have fixed this, and did not

the severe-storm under-read has one obvious explanation. an infrared window
channel sees the top of the cirrus canopy, and over a mature eyewall that canopy
is uniformly cold whatever is happening beneath it - the information is not in
the image to be recovered. at 89 GHz the canopy is transparent and
precipitation-sized ice scatters the signal, so the eyewall prints as a cold
ring around a warm eye. every textbook says this is the answer.

so we went and got it. 456 of the 943 labelled patches have an overpass from
GMI, AMSR2 or one of the three SSMIS flights within 90 minutes, across 100
storms, pulled through OPeNDAP as storm-centred grids of polarisation corrected
temperature. the signal is plainly there: median minimum 155 K against about
270 K ambient, which is deep convection scattering hard.

it does not help.

| model | RMSE | 95% interval |
|---|---|---|
| T3 alone, the served model | 13.48 kt | [11.81, 15.05] |
| T3 refitted on itself, no new data | 13.84 | [12.06, 15.54] |
| T3 + 89 GHz structure | 13.92 | [12.09, 15.63] |
| 89 GHz structure alone | 13.86 | [12.07, 15.57] |

the control is the important row. a correction refitted on the T3 estimate with
no new information at all costs 0.36 kt, so the whole of the apparent damage
from microwave is the refit, and the instrument contributes nothing either way.
the bands it was fetched for did not move: -10.7 to -11.6 at 64-90, -13.8 to
-14.8 at 90+.

we also nearly reported the opposite. restricted to storms whose **truth** is
64 kt or more, adding 89 GHz appears to cut RMSE from 19.6 to 14.6, which reads
like exactly the result we went looking for. it is an artefact. selecting rows
by truth while the model is known to read that band low guarantees the selected
residuals are biased positive, and a model with an intercept banks that whether
or not it is handed any microwave - the control on that same subset scores 13.1,
better still. gated on the **prediction** instead, which is the only thing that
exists at runtime, the gain reverses: 18.3 to 20.1.

what this is not is evidence that microwave cannot help. the literature is
unambiguous that it does, and the reasons ours does not are visible in the
setup: a median 44 minute gap between the pass and the label on a structure that
evolves in minutes, ten hand-cut radial statistics standing in for what is
really a pattern recognition problem, an SSMIS footprint of about 14 km against
a 30 km eye, and 436 joined patches to learn from. tightening the gap does not
rescue it - under 15 minutes it is worse - which points at the features rather
than the collocation.

the honest statement is narrow and it is the one we make: with these features,
this sample and this collocation, 89 GHz did not improve intensity estimation.
a convolutional model on the microwave image itself, trained on a basin's worth
of passes rather than a season's, is the experiment that would settle it, and it
is the first thing to build next. `src/ingest/microwave.py` and
`src/eval_microwave.py` hold all of it, and the imagery is on disk.

## 7c. how small a gain could we even see?

Every improvement we attempted came back null, and for most of the session we
treated that as a series of failed ideas. It is partly something else, and it is
measurable.

Operational verification does not compare two models by printing two confidence
intervals and checking whether they overlap. The National Hurricane Center uses
a two-sided **paired** t-test on a **homogeneous** sample - the same cases for
both models - with the degrees of freedom reduced for serial correlation,
treating forecasts less than 18 hours apart as not independent. Two models
scored on the same patches are not two samples; they are one sample measured
twice, and the storms that are hard for one are hard for the other.

Applying that rule to our own archive:

**943 labelled patches are worth 373 independent observations.** They sit three
hours apart inside storms that last a week, and a cyclone does not reinvent
itself in three hours.

From that, the smallest change we could call real at 80% power and 5%:

| metric | now | smallest detectable improvement |
|---|---|---|
| RMSE | 12.96 kt | 1.78 kt |
| MAE | 9.74 kt | 1.24 kt |
| IMD category exact | 39% | +7 points |
| within 10 kt | 62% | +7 points |

Set the published size of the techniques we did not try against that:

| | typical gain | visible here? |
|---|---|---|
| test-time augmentation | ~0.3 kt | no |
| ensembling several seeds | ~0.4 kt | no |
| a larger backbone | ~0.8 kt | no |
| microwave, done properly | ~2 kt | yes |

So the entire class of dependable small improvements is **below the resolution of
this dataset**. Running them would very likely help a little and we would have no
honest way to say so. That is not a reason to skip them in a system meant to be
used; it is a reason not to claim them here.

Two things follow and both are now in force. **MAE is the more powerful metric
on this sample** - it detects a 1.24 kt change against RMSE's 1.78, because RMSE
is dominated by a handful of tail cases - so it is quoted alongside RMSE rather
than behind it. And **model comparisons use the paired test**, in
`src/eval/paired.py`, not overlapping intervals.

There is a floor underneath all of this as well. Torn and Snyder (2012) put
best-track intensity uncertainty at about 10 kt for tropical storms and 12 kt
for stronger systems **in basins without aircraft reconnaissance**, and the
North Indian Ocean has none at all - IMD's best track is itself largely a
satellite estimate. Our residual shares 58 kt^2 of variance, about 7.6 kt, with
ADT, a completely independent method scored against the same labels. Some of
that is shared infrared blindness rather than label error, so 7.6 kt is an upper
bound, but it is the same order as the published label uncertainty and our total
is 12.96. A meaningful part of what we are measuring is the ruler, not the
model.

## 8. what we don't claim

- we don't beat IMD. official 24 h guidance for this basin is sharper than our
  126 km. the claim is comparable objective guidance in seconds on a laptop,
  with calibrated uncertainty.
- we don't beat ADT either. on IMD best track the two are level within the
  interval; ours is unbiased where ADT reads high, and that is the whole of it.
- the 8.94 kt digital typhoon number is not our T3 result and isn't comparable
  to anything. different basin, sensor and era, and that model served against
  GridSat reported amphan at 0 kt. it's a pretraining stage.
- random split results aren't reported anywhere. a random frame split inflates
  our own 24 h intensity by 10-25%. every figure uses storm or season splits.
- forecast skill at 72 h is positive on this sample but its interval crosses
  zero (track -1.4 to +11.3%), so we don't claim skill that far out. through
  48 h the intervals stay clear of zero.
- the forecast numbers quoted anywhere in this repo are the ones the API serves,
  not the best model in the selection table. they differ: the blend scores
  121.9 km at 24 h, the served ensemble 126.2 km. quoting the better one would
  mean quoting something nobody can call.

## 9. the alerts do not reach the public, by design

we publish CAP 1.2 alerts on an Atom feed, and push them to subscribed systems
over HTTP. we do not send anything to a member of the public, and the gap is
not one we could close by writing more code.

- **it is not ours to issue.** cyclone warnings for the north indian ocean are
  issued by IMD as RSMC New Delhi. NDMA's SACHET carries them to cell
  broadcast, location-based SMS and its own app. a prototype that pushed its
  own cyclone warning to phones would be competing with the warning people are
  supposed to act on, which is worse than useless in an evacuation.
- **the channels are closed to us anyway, and for good reasons.** bulk SMS in
  india needs TRAI DLT registration of the sender and every template; cell
  broadcast needs telco and NDMA access. neither is available to a student
  team, and building a mock of them would only prove we can print a message on
  our own screen.
- so the design goal is to be *consumable* rather than loud: a feed at a stable
  URL, CAP an aggregator already parses, and every alert carrying status
  `Exercise` and a note naming IMD as the real authority. the last mile is
  already built by people with the mandate to run it. what is missing upstream
  is faster objective guidance, and that is what we are.
- **the cadence is measured, not asserted.** `src/replay_alerts.py --suite`
  replays six storms from 25 kt to 130 kt through the live alerting decision
  and writes `reports/alert_cadence.json`. it exists because the first two
  versions of the rule failed it: version one sent on all 25 of amphan's
  forecast cycles, with reasons like a landfall point that "moved" 472 km
  between consecutive cycles and Lhasa entering the threat zone - the 72 h cone
  wobbling, not the storm changing. version two fixed the reasons but still
  sent on all 22 cycles of a 32 kt system that never became a cyclone, because
  the cadence read proximity to a coast before severity.
- the measured result is **2.2 to 4.0 alerts per day** across that range. IMD
  bulletins 3-hourly in the cyclone stage, so 8/day is the operational
  benchmark and every storm here sits below it. the ordering is deliberately
  not monotonic in peak intensity: a landfalling 33 kt depression rates above a
  super cyclone averaged over a life mostly spent at sea, because the tier
  reads proximity as well as strength.
- **that rate is set by the cadence constants, not by the trigger rules.**
  checking which tier applied to each of the 184 alerts, every storm sits at or
  just under its tier's ceiling, and material triggers nearly always coincide
  with a bulletin that was already due. anyone tuning a threshold to change how
  often alerts go out will be disappointed; the constants at the top of
  `api/alert_store.py` are the lever.
- those constants were tuned over four passes against these same six storms.
  there is no held-out set for them and no cross-validation protecting them, so
  they are policy choices rather than measurements, and a seventh storm could
  behave differently.
- what we still do not have: no digital signature on the CAP documents, so a
  consumer cannot verify we sent them. real alerting authorities sign with
  XMLDSig and are listed in the WMO register of alerting authorities. that is a
  registration problem rather than a coding one, but the absence is real and a
  production deployment would need it.
