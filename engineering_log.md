# engineering log

bugs we hit, what caused them, and what caught them.

the thing i'd point at: not one of the serious ones showed up in a loss curve.
every single one was caught by an assertion, a baseline beating the network, or
an end to end run producing something absurd. that's why there's a physics
baseline next to every model and why the pipeline tab runs the whole chain
instead of showing four models separately.

## solved

### the threat zone was drawn 15% too narrow, in the wrong direction

this module exists because the first version of "places at risk" missed kolkata
for amphan with the centre 52 km away. the fix was to widen the cone by the
typical radius of gale-force winds for the forecast intensity. that was right,
and the table it looked the radius up in was wrong.

the table was binned on USA_WIND. that is a 1-minute sustained wind. everything
our models produce is the 3-minute IMD wind, and `gale_radius_km()` is called
with our intensity. a 1-minute wind runs about 11% above a 3-minute one, so a
storm we call 60 kt is about 67 on the scale the table was built on, and the
lookup handed back the band below the right one. every threat zone since has
been about 15% too narrow.

rebuilt in src/wind_radii.py, binned on the same 3-minute wind that looks it up,
and written to reports/wind_radii.json so the table and the lookup can be
checked against each other:

    intensity   old (1-min binned)   new (3-min binned)
    34-48              106 km              120 km
    48-64              130                 153
    64-90              162                 185
    90+                201                 213

it also measures the 50 and 64 kt radii, which nothing had before and which the
probability swaths need.

found by deriving the 50 and 64 kt tables for a different feature and noticing
the 34 kt numbers did not reproduce. the direction matters: too narrow is the
failure mode this whole feature was built to prevent, so it had been quietly
undoing its own purpose.

### giving the intensity model the environment changed nothing, for a reason

T2 and T3 look at a picture and nothing else, which always felt like an omission:
a storm in 20 m/s of shear over a cool sea is not the same storm as one with an
identical cloud top over 30 C water, and T4 has had shear, steering, humidity,
SST and divergence since it was built. so we gave them to T3 as well - the same
ERA5 columns, sampled at each patch's own time and centre, 100% coverage, fused
late so the trunk keeps its Digital Typhoon weights and the environment joins as
numbers at the head.

    image only (serves)    12.96 kt  [11.53, 14.42]   bias -0.09   cat 39%
    image + ERA5           12.88 kt  [11.36, 14.42]   bias -0.68   cat 40%

0.08 kt. the intervals sit on top of each other. not promoted.

the reason is worth more than the experiment. the environment predicts *change*,
not *state*. T3 is a state estimator: it reads one image and says how strong the
storm is now, and how warm the sea is underneath does not change what the storm
is doing at that instant - it changes where the storm is going, which is T4's
job, and T4 already has these exact columns. on top of that the image is not
independent of the environment: a sheared storm looks sheared, that is literally
what the SHEAR scene class is, so handing the CNN a shear number tells it
something it can already see.

so the split we drew between the imagery models and the forecast model turns out
to be the right one, and this is the measurement that says so. the ERA5 sidecar
and the fusion path are kept behind --env, because the same argument predicts it
should help a model of intensity *change*, and that is a different model we have
not built.

### the depressions we miss are not behind the threshold

recall on depressions is 0.39 against 0.76 on named storms, and the obvious
reading is that the detector is simply tuned too conservatively: it is trained
with intensity weighting and its threshold is picked for F1, both of which
favour the storms that matter for a forecast. so give it a second, more eager
operating point and take some of that back.

we did, properly - swept on validation, picked by F2, scored on test, served
point untouched. threshold 0.15. it moves weak-system recall from 0.480 to
0.529 and doubles the false alarms, 0.33 to 0.66 per scene. a fair trade for a
screening view, a bad one for anything that becomes a forecast, which is why the
tier is now explicit rather than a global setting.

the useful part was the control we almost did not run. take the threshold all
the way down to 0.05:

    thr 0.05   precision 0.204   weak recall 0.595   2.73 false alarms per scene

eight times the false alarm rate of the served point, and four in ten weak
systems are still missed. that kills the hypothesis. these are not systems the
network saw faintly and rejected - if they were, they would appear as the
threshold fell. they are systems it does not see, which is exactly what you
would expect from a 20 kt depression in an infrared image, where there is no
organised convective signature to find and the thing looks like ordinary
monsoon cloud.

worth stating plainly because it changes what the limitation means. "we miss 6
in 10 depressions" sounds like a model we undertrained. "we miss 6 in 10
depressions and cannot find them at any sensitivity we can afford" is a
statement about infrared imagery, and the fix is a different instrument rather
than a better threshold.

### the INSAT scene gap is the sensor, and it is two classes

scene typing reads 0.72 on GridSat and 0.60 on INSAT, and INSAT is the live
feed, so this was the first thing worth attacking with eleven days left. it did
not work, but the diagnosis is worth more than the fix would have been.

the lead was real. the cold-cloud half of the hybrid is a logistic model on
brightness-temperature statistics, fitted with one shared scaler and no sensor
term, and the two sensors' features are not on the same scale: the
water-vapour spread differs by 0.9 standard deviations, the mean by 0.7, the
coldest pixel by 0.67. one model cannot fit both. fitting it per sensor lifts
that half from 0.415 to 0.505 on INSAT, which is a large move.

it buys almost nothing. the hybrid goes 0.601 to 0.608, because the CNN
dominates the blend and the CNN is the half that is actually weak. a per-sensor
blend weight was worse than the fixed 0.5 - the curve already peaks there on
INSAT, and choosing per fold just adds selection noise (0.608 to 0.605).

so we went at the CNN. if the sensors differ by an offset, standardising each
patch by its own channel statistics should remove it. fifteen minutes of
cross-validation says no, and says it clearly: 0.722 to 0.703 on GridSat and
0.601 to 0.560 on INSAT. worse on both. the reason is the point of the whole
entry - the offset it removes is absolute brightness temperature, and the ADT
scene classes are partly *defined* by absolute brightness temperature.

which pointed at the test we should have run first. all 505 INSAT patches have
GridSat twins at the same storm and time with the same label, so we can score
the same scenes through both instruments and take scene composition out of it
entirely. the gap survives - 0.725 against 0.601 - and the per-class split is
the answer:

    EYE     0.76 -> 0.75      SHEAR  0.84 -> 0.76
    CRVBND  0.75 -> 0.64      EMBC   0.69 -> 0.47      IRRCDO 0.58 -> 0.39

an eye is a hole in the cloud. it is still a hole in the cloud on a different
satellite. whether a central dense overcast is "embedded" or "irregular" is a
judgement about how cold and how uniform it is, and that does not survive a
change of instrument, calibration and viewing angle. the two classes that
collapse are exactly the two with no geometry to hold on to.

four attempts and no fix, so nothing was promoted and the GridSat-trained
hybrid still serves. what changed is what the system says: `/scene` now returns
the measured F1 for the class it just predicted on the sensor it just read, so
an EMBC off the live feed arrives with "0.47 here, 0.69 on GridSat" attached.
the honest version of an unfixed limitation is a number next to the answer, not
a sentence in a document nobody opens.

### auditing the deck against the reports found five stale numbers and a bug

before freezing the deck i checked every hard-coded figure in it against
`src/eval/headline_numbers.py`, which is supposed to be the arbiter. the audit
found more in the arbiter than in the deck.

**the arbiter was reporting a model that no longer serves.** its T3 block read
`cv/t3_best_track_rotate.json`, the GridSat-only run, while the API serves the
jointly trained model from `cv/t3_best_track_rotate_joint.json`. so the script
that documents disagreed with the thing it documents: 13.16 kt against 12.96,
and a different band table. same for T1, where it printed only the GridSat
detector, making INSAT detection look like 0.41 F1 when the checkpoint that
actually answers for INSAT imagery gets 0.68. both now print what serves, with
the alternative beside it.

**a calibration cost that matched no run.** the deck, limitations and this log
all said recalibration costs 2.7 kt of RMSE, 13.2 to 15.9. the report says 12.96
to 15.32, a cost of 2.4. the 15.9 came from a run that predates the best-track
crosswalk and had been copied forward three times without anyone re-reading the
file it came from. the deck now computes the difference from the report rather
than restating it.

**a storm count nobody could reproduce.** the deck said 4.4 named storms a
season. IBTrACS over 1990-2025 gives 167 storms in 36 seasons, which is 4.6, and
no definition i tried produced 4.4. the basin's own counts are now printed by
the arbiter, read from the best track file itself, so the claim is checkable
like any other.

**timings that were all wrong in the same direction.** API.md promised a
forecast in 3.3 s, a scene in 1.3 s and the pipeline in 30 s. measured: 1.5 s,
0.3 s, 14 s. nothing was slower than claimed, which is why it survived - an
overstatement of your own latency never produces a complaint.

**the live feed is 3-hourly, not hourly.** `wanted_slots` steps back in three
hour synoptic slots and always did; the deck had said hourly. at any moment the
newest slot is 1 to 4 h old, which is what it says now.

and the bug: `src/eval/numbers.py` shadowed the standard library's `numbers`
module. numpy imports `numbers` internally, so any script in `src/eval/` that
imported numpy or pandas crashed on import with a circular-import error from
numpy - `src/eval/t2_hybrid.py`, the script that produced the T2 numbers, could
not be run from a clean checkout at all. renamed to `headline_numbers.py`.

then the same audit on the three documents we brief ourselves from, and that
was worse. the demo script, the panel brief and the primer had all been written
before the best-track crosswalk and none had been revisited. between them they
would have had us say out loud: T2 beats its baseline 0.689 to 0.563 (the
cross-validation says 0.664 to 0.650, a tie, which is the whole reason the
hybrid exists), intensity 10.95 kt against a 21.06 physics baseline (both
against ADT's wind, not best track), 121.9 km and +14.2% at 24 h (the selection
table's blend, which nothing calls), and landfall position 257 km with "adding a
land mask is the known fix" (it was added; it is 184 km). the primer taught the
recalibration as applied and served, when the cross-validation is what took it
out of service.

worse than the deck, because a slide gets read and a script gets spoken. all
three now carry what the reports say, and the demo's INSAT beat - one storm,
five scenes, a terminal script - is now the Live tab.

the lesson is the one already at the bottom of this log, arrived at from the
other direction: a number that is restated rather than read goes stale silently,
and the file it came from is the only thing that can catch it.

### the alerting rule sent on every forecast cycle, three times over

we could render a CAP document but had no way to publish one: `/cap.xml` built
fresh XML per request, stamped a new identifier each time, always said
`msgType=Alert`, and stored nothing. so nothing could subscribe. the fix is an
alert store with a lifecycle - Alert, then Update carrying `<references>` to
what it supersedes, then Cancel - behind an Atom feed, which is how CAP
actually travels.

the interesting part is the decision of *when* to send, because it failed three
times and each failure was only visible by replaying real storms through it.
`src/replay_alerts.py` does that, fix by fix, and prints the reason for every
send. the reason strings are the whole diagnostic; none of this was visible by
reading the rule.

**v1 - thresholds only.** re-issue when the picture changes by more than half
our own measured error (landfall timing MAE 9.0 h, position 114 km). sounded
principled. sent on **25 of Amphan's 25 cycles**, with reasons like a landfall
point that "moved" 472 km between consecutive cycles and Lhasa entering the
threat zone. the threshold was right and the quantity was wrong: cycle-to-cycle
jitter in the landfall point is far larger than our error against truth, and a
72 h cone sweeping the Himalayas picks up new towns forever.

**v2 - gate on what the forecast can see.** landfall shifts only count inside
48 h, a place only counts inside 36 h and is announced once per storm, upgrades
interrupt but downgrades ride the next routine bulletin, and a routine cadence
underneath. Amphan dropped to 23/25 with clean reasons - but a 33 kt monsoon
depression still sent on **all 22 of its cycles**, because the cadence read
proximity to a coast before severity. a depression near land is not a super
cyclone near land, and India gets many depressions.

**v3 - severity before proximity, and a stated rate limit.** cadence keys off
intensity first; the intensity trigger compares against the strongest the storm
has *ever* been, so a deep depression oscillating back and forth stops
announcing that it has intensified for the third time; the landfall-shift
trigger is off below cyclone strength, where the landfall point is the least
reliable thing we produce; and under it all a minimum gap that only the four
things you would wake someone for may break.

**v4 - the measurement was wrong too.** the suite reported alerts per day as
`3 h x cycles`, assuming every IBTrACS fix is 3-hourly. they are 6-hourly here,
so every rate it had ever printed was exactly double. the span now comes from
the timestamps. "share of cycles" was dropped as a headline at the same time:
it is really a statement about how densely the archive sampled a storm, not
about how much the system talks.

and one attempted fix taught more by failing. suppressing the place trigger
once the centre is inland and below cyclone strength - the right rule, since
what people need then is a rainfall warning, which is IMD's product and not
ours - moved 19 alerts out of "new place threatened" and changed the total by
nothing at all. every storm's count came back identical.

that is the real finding: **the alert rate is cadence-bound, not
trigger-bound.** checking which tier applied to each of the 184 alerts, every
storm sits at or just under its tier's ceiling - the 25 kt system spent all 21
of its alerts on the 12 h watch cadence and measured 2.2/day against a 2.0
ceiling; the 33 kt system sat 18 of 21 on the 6 h cadence and measured 4.0
against 4.0. material triggers almost always coincide with a bulletin that was
due anyway rather than adding to it. so the cadence constants are the policy,
and the trigger rules decide what an alert *says*, not how often one goes out.
worth knowing before anyone tunes a threshold hoping to change the rate.

the rates land at 2.2-4.0 per day across 25 kt to 130 kt, in
`reports/alert_cadence.json`, regenerated by `src/replay_alerts.py --suite`.
IMD itself bulletins 3-hourly in the cyclone stage, so 8/day is the
operational benchmark and everything here sits below it. the ordering is not
monotonic in peak intensity and should not be: a landfalling 33 kt depression
rates above a super cyclone averaged over a life mostly spent at sea, because
the tier reads proximity as well as strength.

the honest caveat: four passes of this were tuned while looking at the same six
storms, so the constants are fitted to them in a way no cross-validation
protects against. they are policy choices, not measurements, and they are
written as named constants at the top of `api/alert_store.py` so that is
visible rather than buried in a condition.

a related claim died on the way: API.md said the CAP was "validated against the
OASIS schema", and when `<references>` had to be added there was nothing in the
repo that could check it. no XSD, no validator, nothing that ever ran.
`src/check_cap.py` is the check that claim needed - element order, closed value
lists, and a hard refusal of any status but `Exercise`. its own first version
passed a deliberately invalid `<severity>` because it only looked at children
of `<alert>` and severity lives inside `<info>`; a negative test caught that,
which is the only reason the checker is worth anything.

### T3 was graded against ADT, not against best track

found while writing out what "ADT labels" means. `patches.csv` took `vmax_kt`
straight from the ADT archive and the trainer's own comment called it
"best-track intensity". it wasn't. every T3 number we published, 10.95 kt
included, measured how closely the model copies ADT.

that matters because the two are not the same wind. ADT follows the dvorak
tables, which give a 1-minute mean; IMD's best track is 3-minute. matched in
space and time across 11,227 fixes, ADT reads +7.6 kt above IMD, a ratio of
1.106 at 34 kt and up. so T3 learned ADT's scale, the dashboard mapped it onto
IMD's category bands, and the imagery pipeline fed it to T4, which was trained
on IMD winds.

the fix is a spatio-temporal crosswalk: interpolate every best track to the ADT
analysis time and take the nearest storm within 200 km. median separation 28 km,
no ADT storm splitting across two SIDs, and 943 of 974 patches matched against
728 that had an ADT wind - the correct target also turned out to be the more
plentiful one.

5-fold cross-validation by storm, same 719 scenes, everything against IMD best
track:

| | RMSE | bias |
|---|---|---|
| what we were serving (ADT target, recalibrated) | 19.0 kt | +7.4 |
| ADT itself | 13.95 kt | +7.1 |
| retrained on best track, no recalibration | 13.5 kt | +0.2 |

the old model's single test split had said 13.8 kt. cross-validation says that
split was a lucky draw.

### the recalibration that fixed the bias made the error worse

inverting the fitted line does what it promised - the 90 kt+ band goes from
-13 kt to +2 - but under cross-validation it costs 2.4 kt of RMSE (13.0 to
15.3), and that holds even when the line is fitted on four folds of held-out
predictions at once rather than one small validation set. expanding the spread
fixes the average and adds variance.

(those four figures are re-measured on the model that serves now, the one
trained on both sensors. the entry first said 2.7 kt, 13.2 to 15.9, which was
a number carried from an older run and matched no report file - found while
checking the deck against `src/eval/headline_numbers.py` before submission.)

so it isn't served any more. the severe-storm under-read goes back into
limitations with its measured size instead of being papered over.

### "mirror plus a half turn keeps the storm cyclonic" doesn't

three datasets augmented with `rot90(a[:, ::-1], 2)` under a comment claiming
the half turn undid the mirror. a mirror followed by a half turn is a vertical
flip, and every reflection reverses the sense of rotation. checked it on a 3x3
array: clockwise in, anticlockwise out. half of T2 and T3's training storms were
spinning the southern hemisphere way.

rotation-only augmentation won for T3 on every metric under cross-validation
(13.16 vs 13.38 kt) and lost slightly for T2 (0.664 vs 0.638 macro-F1 for the
CNN), which makes sense: a cloud pattern has no handedness to lose. each task
now uses what its own cross-validation picked.

### the single split flattered T2, and the physics baseline was never beaten

cross-validated over all 974 patches the CNN scores macro-F1 0.664 and the
cold-cloud statistics baseline 0.650 - the same, inside the intervals. the old
single split had said 0.689 against 0.563.

in hindsight it is not surprising. ADT assigns scene type with rules on
cloud-top temperature, and those statistics measure exactly that. we were asking
a CNN to beat the thing the labels were made of.

what saved it is that they fail on different scenes: the CNN is far better on
EYE, the statistics on SHEAR and IRRCDO. averaging the two sets of probabilities
50/50 gives 0.710, better than either alone by 0.029 to 0.065 macro-F1 with 95%
confidence. that hybrid is what serves now. equal weights, so nothing was tuned
on the held-out folds.

### the INSAT "strips" were our own subsampling

we had written that an INSAT granule is a latitude strip that moves between
campaigns. the strips are real - they are the rapid-scan sectors ISRO runs over
an active storm - but the full sector is still there every half hour. 3DR scans
it at :15 and :45, 3D and 3DS at :00 and :30, each covering 9.5S to 43.6N.
our nearest-to-the-hour rule picked the :02 rapid scan every single time.

measured on storm days in 2019, 2020 and 2024 and a quiet day in 2017 before
changing anything. selecting on the minute gets whole-basin imagery, which is
what made an INSAT archive worth downloading at all.

### MOSDAC access tokens last about five minutes

the first bulk download died at file 117 of 879 with INVALID_TOKEN. MOSDAC's own
client swaps a refresh token for a new pair when that happens, and a refresh
needs no password, so it cannot count toward the three-strikes lockout. ours does
the same now, logs out at the end, and retries dropped connections - which are
not auth failures - three times before giving up. 10.2 GB then came down in one
run with three refreshes and nothing lost.

### the map drew india's boundary the way OpenStreetMap does

OSM maps boundaries by who administers the ground, which is not how the
government of india depicts its own, and this is a screen meant for a MoES
panel. replaced with natural earth's india point-of-view edition, vendored into
the repo, and the ingest asserts gilgit, aksai chin and tawang all fall inside
india's polygon before writing the file. the map now needs no network at all.

### the cone missed kolkata with amphan 52 km away

the first version of "places at risk" used the forecast cone alone. a cone says
where the centre might go; it says nothing about how wide the storm is. widened
by the typical radius of gale-force winds for the forecast intensity - measured
from JTWC wind radii in IBTrACS - and kolkata appears, at 0 h, which is where
it was. (that table was 106 km at 34-48 kt rising to 201 above 90. it was also
binned on the wrong wind scale; see below. it is now 120 rising to 213.)

### "stay at sea for 72 h" for a storm already ashore

the landfall model answers whether landfall is still to come, which is false
once it has happened, so amphan at 20 may 12 UTC came back with a 4% landfall
probability and a CAP headline saying it would stay at sea. IBTrACS puts
distance to land at 0 over land; the API now says the centre is inland instead.

### a grad-CAM hook fired inside somebody else's forward pass

the imagery tab asks for an intensity estimate and its attention map at the same
moment. FastAPI runs those on a thread pool, both threads reach the same model
object, and the explain hook - which calls `retain_grad` - fired inside the
other thread's `no_grad` pass and crashed it. one lock around every model call,
and the lazy loaders moved under it too: they set "loaded" before loading
finished, so a second request mid-load got `None` and a 404.

### the patch cache rebuilt itself on every run

five of the 979 labelled scenes always sit too close to the crop edge to cut a
patch from, so the cache holds 974. the rebuild check compared those two numbers
and concluded imagery was missing, every time - twelve minutes of rebuilding,
and it silently dropped the best-track columns T3 now trains on. it compares
against what the last build actually saw now, and re-adds the columns itself if
it does rebuild.


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

both numbers there are against ADT's wind, and the fix did not survive being
cross-validated: see the two entries at the top of this log. the shrinkage is
real, the inversion trades 2.4 kt of RMSE for it, and it is no longer served.

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
- intensity reads the strongest storms low: -8 kt at 64-90, -13 at 90+
- landfall position is 184 km, better than 256 but not evacuation grade
- RI is barely skilful at +0.096
- IRRCDO is the weakest scene class, F1 0.53 on 97 patches
- 72 h forecast skill is positive but its interval crosses zero
- test sets are small, the basin makes about 5 storms a year

## what judges will probably ask

**why not INSAT for an indian problem statement.** we do use it. everything was
trained on GridSat first, because it needs no approval and a system that depends
on one you don't control isn't a system. MOSDAC access then came through and the
INSAT-3D and 3DR archive is mirrored onto the same grid at the same slots, so
the two sensors can be compared on identical storms, and the live feed is
INSAT-3DS.

**do you beat IMD.** no, and we don't claim to. IMD runs multi model numerical
guidance we have no access to. we beat CLIPER, which is the statistical benchmark
operational centres score skill against, by 11% on track and 20% on intensity at
24 h, with intervals that stay clear of zero out to 48 h.

**your dvorak labels are machine generated.** correct, they're from CIMSS ADT, so
T2 measures agreement with ADT rather than with an analyst. it also explains why
a logistic model on cloud-top temperature statistics matches the CNN: ADT assigns
scene type from rules on those same temperatures. serving the average of the two
is the honest way to use that.

**what is your ground truth for intensity.** IMD's best track, interpolated to
the scene time. it was ADT's own estimate until we caught it - the entry at the
top of this log - and ADT runs 7.6 kt above IMD because it reports a 1-minute
wind. against best track, our estimate and ADT's are level (13.2 vs 13.9 kt RMSE,
difference -2.3 to +1.5) and ours is unbiased where ADT reads high.

**8 km is too coarse for dvorak.** it costs us eye detail. but we tested the
claim instead of assuming it and it doesn't hold as a ceiling: within 64 kt+ the
model correlates 0.883 with truth and reproduces its spread.

**could this run tomorrow.** it runs today. the live tab pulls the latest
INSAT-3DS full-sector scan from MOSDAC, puts it on the model grid and runs the
same chain on it; the imagery is about an hour old, which is MOSDAC's publishing
lag. the archive models are unchanged - only the loader knows the difference.

**your test sets are small.** they are, and that's the basin: about five named
storms a year. so nothing rests on one split any more. T2 and T3 are scored by
5-fold cross-validation grouped by storm - every patch predicted by a model that
never saw its storm - and every headline number carries a 95% interval from
resampling whole storms. that is also how we found that our single test split
had been flattering both of them.
