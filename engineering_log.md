# engineering log

bugs we hit, what caused them, and what caught them.

the thing i'd point at: not one of the serious ones showed up in a loss curve.
every single one was caught by an assertion, a baseline beating the network, or
an end to end run producing something absurd. that's why there's a physics
baseline next to every model and why the pipeline tab runs the whole chain
instead of showing four models separately.

## solved

### detection trained on zero positive pixels

focal loss identifies a centre by the target being exactly 1.0. we built targets
by sampling a gaussian at cell centres, and the storm centre almost never lands
exactly on one, so the peak came out at 0.98 or 0.97. never 1.0.

so there were no positive pixels anywhere in the training set. the loss was
well defined and went down smoothly, because the model was being asked to learn
one consistent thing (there are no storms) and it learned that perfectly.

fix is one line, pin the cell nearest the true centre to exactly 1.0. a loss
curve cannot tell you your targets are empty.

### the ADT parser quietly deleted 89% of eye scenes

the ADT archive is fixed width text across three format generations. our first
parser split on whitespace and read fields by position.

the "estimated radius of max wind" field is `N/A` (one token) for most scenes
and `8 IR` (two tokens) for others. the two token form happens exactly when
there's an eye. so every eye scene's columns shifted by one and got dropped.

caught by an assertion that parsed coordinates have to be inside the basin box.
6025 records failed it. re-anchoring the parse on the geographic fields took EYE
from 125 examples to 1203.

### convnext wouldn't train at 3e-4, and the baseline is what noticed

T3 sat at 17.08 kt validation RMSE. the validation set's own standard deviation
is 17.1, which means it had collapsed to predicting the mean.

the loss curve looked unremarkable. the tell was that the cold cloud physics
baseline was beating the network, 15.73 vs 17.08. a network that can't beat a
threshold on pixel counts hasn't learned anything.

isolated it by trying to deliberately overfit 100 samples. a healthy net drives
that to near zero and ours wouldn't. at 5e-5 it reaches 0.015.

3e-4 is a fine default for training from scratch and far too large for fine
tuning a pretrained backbone, the first few updates overwrite the pretrained
weights. T2 shared the same backbone and the same learning rate and would have
failed identically without ever looking obviously broken.

also found a missing ImageNet input normalisation while in there.

### the right model in the wrong imagery domain

T3 was trained on digital typhoon PNGs and served against GridSat IR/WV. the
pipeline reported amphan, a 120 kt storm, at 0 kt.

different sensor, different channels, different hemisphere. fixed by making
digital typhoon a pretraining stage and fine tuning on GridSat
(`train_intensity_gridsat.py`). the 8.94 kt digital typhoon number is not
comparable to anything and isn't claimed anywhere.

### intensity regressed to the mean and one number hid it

model fit was `predicted = 0.852 x truth + 7.60`. slope under 1 means everything
gets squeezed toward the middle: weak storms too strong, severe too weak.

the two errors have opposite signs so the overall bias came out at -0.5 kt,
which looks excellent and conceals both. we only found it by reporting bands.

fixed by inverting the fitted line, fitted on validation and applied unchanged
to test. RMSE 11.66 to 10.95 and every band improved.

worth noting: the docstring in `calibrate_intensity.py` said 0.727 / 13.9 for a
while, from an older run, while the checkpoint and the report both said
0.852 / 7.60. i'd quoted the stale one in a doc. always trust the report file,
the coefficients get refitted every run.

### depressions were drowning the detection loss

the basin has far more weak depressions than severe cyclones, so the loss was
dominated by systems nobody evacuates for.

added a per storm weight rising with intensity,
`w = clip((vmax/34)^1.5, 0.3, 3.0)`. 34 kt is the naming threshold so a cyclonic
storm sits at 1.0.

F1 0.518 to 0.586, recall 0.437 to 0.540, precision held. every band improved
including depressions, even weighted down to 0.3x. teaching the net what an
organised storm looks like made it better at marginal ones too.

### a weighting scheme that could only weight upward

first attempt at the above did nothing. it combined overlapping weights with
`np.maximum` against a background of 1.0, and any downweight is below 1.0, so
the maximum always threw it away.

fixed with an influence map where the nearest storm wins each cell.

### two evaluation errors of my own

**misaligned verification.** i scored pipeline forecasts against valid times up
to 15 h adrift and concluded 3 of 4 verifying positions fell outside their cone,
and that our uncertainty was understated. reported it before catching it.
corrected: 11 of 12 inside.

**misread column.** reported "amphan improved from -26 to -10 kt" after reading
a 6 hour forecast row as the detected intensity. the real comparison was 74 to
71 kt, slightly worse.

both were confident and wrong. verification code deserves the same suspicion as
model code.

### seven hours of training with the GPU at 0%

per sample netCDF reads at 683 ms each. the GPU sat idle waiting on disk.

fixed with a uint8 memmap cache, 683 ms to 53 ms. also found `evaluate()` was
reopening the netCDF twice per scene just to read constant lat/lon arrays, about
8 minutes wasted per pass.

### the dashboard died silently with no internet

maplibre was loading from unpkg. when the CDN failed the page went blank with no
error at all.

i first blamed browser flakiness, which was wrong. fixed by vendoring maplibre
into `web/vendor/`, adding a painted background layer so losing the tile server
costs coastlines not the storm, and an `isStyleLoaded()` guard because maplibre
won't re-fire `load` if the style finished before the handler attached.

later dropped google fonts for the same reason. the dashboard now needs the
network only for basemap tiles.

### splits that reshuffled themselves as data arrived

T2 looked like it regressed from 0.723 to 0.631 and i spent real time hunting a
bug that didn't exist.

the split was a seeded shuffle of the storm list, and GridSat was downloading
incrementally, so the list grew between runs and every storm got reassigned.
splits are hashed on storm id now, so adding data enlarges them without
reshuffling.

the honest number is the lower one.

### cross basin contamination in the best track

the IBTrACS "north indian" file has 25 western pacific typhoons in it, all
entirely east of 100E, because IBTrACS tracks storms across basins.

they were training a north indian model on a different basin's dynamics.
filtering took 291 storms to 266 and moved RI skill from +0.055 to +0.096.

### the API broke on a change i made to the trainer

added an intensity weight map to the detection dataset, so
`BasinScenes.__getitem__` started returning four values instead of three.
updated the training script, forgot the API, which still did
`x, _, _ = ds_wrap[0]`.

the pipeline endpoint 500'd with "too many values to unpack". it had been broken
for a while and i missed it because i was verifying T1 from the report JSON
rather than through the API. fixed by indexing instead of destructuring so a
fifth return value can't do it again.

### INSAT fill values map to the coldest temperature

count 1023 is no-data, and the lookup table is inverted, LUT[0] is 340 K and
LUT[1023] is 179.9 K. so fill converts to the coldest possible cloud top, and
about 70% of a granule is fill.

unmasked, the detector would see a basin sized sheet of deep convection and find
storms everywhere, with a completely healthy loss. caught by checking a known
clear sky point in the arabian sea (299 K, correct) instead of trusting the
sector mean, which was suspiciously cold at 206 K.

### smaller ones

- **RI brier skill of -1.86** from 30x positive class weighting. it bought
  recall by inflating every probability. fixed by training unweighted, fitting
  a platt scaler on validation, and moving the decision threshold instead.
  went to +0.096.
- **NaN at epoch 11**, focal loss in fp16 under autocast. forced that bit to
  fp32.
- **decoder at stride 8 against stride 4 targets.** shapes were compatible
  enough that nothing crashed, the loss just compared the wrong cells and the
  model learned a blurred offset centre. silent shape mismatches are the worst
  class of vision bug because every symptom looks like underfitting.
- **negative wind in the bulletin**, `25 kt (-0-50)`. clamped the band at 0.
- **plain GBM lost to CLIPER on track** at every horizon, -2.6% at 24 h.
  displacement is near linear in the persistence predictors and a tree ensemble
  overfits 4.4k rows. fixed by boosting on CLIPER's out of fold residuals
  instead, so the learned part can only add a correction and degrades to CLIPER
  where there's nothing to add.
- **MSYS2 python venv unusable for pytorch**, had to use native cpython.
- **UTF-8 BOM broke `.cdsapirc`.**

## still open

see limitations.md, it has the numbers. short list:

- detection misses about 6 in 10 depressions
- severe storm intensity still reads ~12 kt low at 64-89 kt
- landfall position is 184 km, better than 256 but not evacuation grade
- RI is barely skilful at +0.096
- IRRCDO has 7 test examples so it's not really measurable
- test sets are small, the basin makes about 5 storms a year

## what judges will probably ask

**why not INSAT-3D for an indian problem statement.** we asked, approval came
late, so we designed around it. GridSat carries the same IR window channel
globally and our models take a generic 3 channel patch, so swapping is a data
loader change plus a fine tune. building on an approval we didn't control would
have been the actual mistake.

**do you beat IMD.** no, and we don't claim to. IMD runs multi model numerical
guidance we have no access to. we beat CLIPER, which is the statistical
benchmark operational centres score skill against, by 14% on track and 23% on
intensity at 24 h.

**your dvorak labels are machine generated.** correct, they're from CIMSS ADT.
so T2 measures agreement with ADT, not with a human analyst. it's in the
limitations.

**8 km is too coarse for dvorak.** it costs us eye detail. but we tested the
claim instead of assuming it and it doesn't hold as a ceiling, within 64 kt+ the
model correlates 0.883 with truth and reproduces its spread. the severe storm
error was calibration, not resolution.

**could this run tomorrow.** not as it stands, and it's a data problem not a
model one. GridSat is a delayed archive. live needs a real time feed, INSAT
through MOSDAC or himawari or meteosat. the pipeline and models don't change.

**your test sets are small.** they are, and that's the basin. the north indian
ocean makes about five named storms a year. we hold out 51 storms, 354 detection
scenes, 974 scene patches, 728 intensity patches, and report sample counts for
every band so nobody over-reads a thin one.
