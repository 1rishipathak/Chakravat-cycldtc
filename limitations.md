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

what we tried: extending the archive from 4 seasons to 14 (224 to 921 scenes)
moved F1 only 0.461 to 0.518. poor return on 4x the data, which was good
evidence the limit wasn't sample count.

what worked was changing what the loss cares about. each storm now gets weighted
by intensity, `w = clip((vmax/34)^1.5, 0.3, 3.0)`. F1 went 0.518 to 0.586,
recall 0.437 to 0.540, and every band improved including depressions, even
though depressions got weighted down to 0.3x.

the 0.90 target isn't reachable on 8 km imagery and more of the same data won't
get there.

## 2. intensity reads low on severe storms

worst limitation operationally, because under-reading a severe cyclone is the
wrong direction to be wrong in.

| band | n | bias before | bias after |
|---|---|---|---|
| under 34 kt | 24 | +5.3 | +2.7 |
| 34-47 | 32 | +4.8 | +3.4 |
| 48-63 | 18 | -0.9 | -0.5 |
| 64-89 | 12 | -14.2 | -12.0 |
| 90+ | 10 | -16.6 | -7.0 |

overall RMSE 10.95 kt, MAE 7.36, bias -0.51. the overall bias looks great and
hides the table completely, the errors are opposite signed and cancel. that's
why we report bands.

### a hypothesis we published and then disproved

an earlier version of this file argued the residual error wasn't a modelling
problem at all. eye diameter and eyewall gradient are at or below the 8 km pixel
scale, so a model can't read detail the sensor never recorded. we stopped tuning
on the strength of that.

it was wrong. within the 64 kt+ band the model correlates +0.883 with truth and
reproduces the spread almost exactly (26.2 kt predicted vs 25.1 actual). a model
that couldn't resolve intensity would show neither. the information was there,
the mapping was broken.

fitted on validation:

```
predicted = 0.852 x truth + 7.60
```

slope under 1 is regression toward the mean under squared error loss. weak
systems come out too strong, severe ones too weak, and they cancel into that
healthy looking overall bias. inverting the line took RMSE 11.66 to 10.95 and
improved bias in every band, most at 90+ kt.

two caveats we're keeping. the 64 kt+ band is 22 test patches, so this is
evidence against our own hypothesis rather than a precise measurement. and
64-89 kt is still 12 kt low, so it's reduced not solved.

the case for INSAT is now narrower and more honest. 4 km resolves eye structure
GridSat can't and we expect it to help, but we can't claim resolution was the
binding constraint because we tested that and it wasn't.

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

## 5. the dvorak labels are algorithm output

scene labels come from the CIMSS ADT archive, 8212 labelled north indian scenes
2003-2025. a model trained on them learns to reproduce ADT, not a human
forecaster.

we think that's the right target since the problem statement asks for objective
automation of dvorak and ADT is the operational objective standard. but it's a
different claim from "matches an expert" and we don't make the second one.

current: macro-F1 0.689, accuracy 0.764, vs 0.563 for cold cloud stats and
0.140 for majority class. per class SHEAR 0.867, CRVBND 0.819, EYE 0.788,
EMBC 0.636, IRRCDO 0.333.

test set is 123 patches so these have wide error bars. IRRCDO is 7 test
examples and basically not learned, it's the rarest class and the most visually
ambiguous.

an earlier run said 0.723 and it's tempting to quote that instead. it's not
comparable, the split was a seeded shuffle and the storm list grew between runs
so every storm got reassigned. splits are hashed now. the lower number is the
honest one.

## 6. the pipeline inherits its own upstream errors

run from imagery alone the chain detects storms within 47-94 km and puts 11 of
12 forecast positions inside their stated cone on amphan, tauktae and biparjoy.
but the forecast is driven by the detected track, not best track, so detection
scatter propagates into the motion estimate.

two features have no imagery equivalent, central pressure and distance to land.
they get passed as missing rather than guessed. the boosted trees handle NaN
natively. guessing would have hidden the gap.

## 7. imagery is coarser than operational

GridSat is 8 km 3-hourly, INSAT-3DR is 4 km half-hourly. everything here was
trained on the coarser one because it needs no authentication.

MOSDAC approval has since come through. every number in this file was still
produced without INSAT. 

expected gain, stated up front so it can be checked: most on limitation 1, where
4 km should help small weak systems. least on limitation 2, because section 2
above shows resolution wasn't the binding constraint there.

## 8. what we don't claim

- we don't beat IMD. official 24 h guidance for this basin is sharper than our
  121.9 km. the claim is comparable objective guidance in seconds on a laptop,
  with calibrated uncertainty.
- the 8.94 kt digital typhoon number is not our T3 result and isn't comparable
  to anything. different basin, sensor and era, and that model served against
  GridSat reported amphan at 0 kt. it's a pretraining stage. our number is
  10.95 kt.
- random split results aren't reported anywhere. a random frame split inflates
  our own 24 h intensity by 10-25%. every figure uses storm or season splits.
- this isn't real time. GridSat is a delayed archive product. live operation
  needs a real time feed, the models don't change.
