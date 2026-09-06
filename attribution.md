# data sources

this project trains on public scientific datasets. some of them require
attribution to use, and ERA5 has a licence line that has to be copied word for
word, so it's all collected here.

nothing is redistributed in this repo. `.gitignore` excludes `data/` except the
clipped coastline. every source below gets fetched by a script in `src/ingest/`
from its own provider, under that provider's terms.

## imagery

**GridSat-B1**, NOAA NCEI. global geostationary IR and water vapour brightness
temperature, 0.07 degrees, 3-hourly, 1980 to now. US government work so public
domain, citation requested not required.

> Knapp, K. R., et al. Globally gridded satellite observations for climate
> studies. Bulletin of the American Meteorological Society, 2011.

**INSAT-3DR**, ISRO, distributed through MOSDAC (space applications centre,
ahmedabad). used under the MOSDAC data policy with a registered account.
granules are per-user and are not redistributed here, `src/ingest/insat.py`
fetches them with your own credentials.

**digital typhoon**, national institute of informatics, japan (kitamoto et al).
released under creative commons attribution 4.0, so attribution is required.
only used to pretrain the intensity trunk before GridSat fine tuning.

## best track and reanalysis

**IBTrACS v04r01**, NOAA NCEI. our ground truth for position and intensity.

> Knapp, K. R., et al. The International Best Track Archive for Climate
> Stewardship (IBTrACS). Bulletin of the American Meteorological Society, 2010.

**ERA5**, copernicus climate change service (C3S) climate data store, ECMWF.
gives us the environmental predictors: shear, mid level humidity, SST and
steering flow.

the copernicus licence requires this, verbatim:

> Contains modified Copernicus Climate Change Service information.
>
> Neither the European Commission nor ECMWF is responsible for any use that may
> be made of the Copernicus information or data it contains.

## labels

**CIMSS advanced dvorak technique archive**, university of wisconsin-madison.
source of the 8212 dvorak scene labels behind T2.

these are algorithm output, not analyst judgement. a model trained on them
learns to reproduce ADT rather than a human forecaster. that's in
limitations.md section 5 and it's a real ceiling on what the T2 number means.

## map and interface

**natural earth** 1:10m coastline, public domain. clipped to the basin and
vendored at `data/static/coastline_nio.geojson` because the landfall module
needs it at runtime and a demo shouldn't depend on a network.

**maplibre GL JS**, BSD-3-clause, vendored at `web/vendor/`.

**openstreetmap** basemap tiles, © openstreetmap contributors, under ODbL.
tiles are fetched live and not redistributed. the dashboard paints a solid
background under them so losing the tile server costs the coastlines but not
the storm.

## what this isn't

chakravat is a prototype for SIH problem statement 26070. it is not an IMD
product, not affiliated with IMD or ISRO, and must not be used for operational
warning. official tropical cyclone warnings for the north indian ocean come
from RSMC new delhi.

the interface follows government of india design convention because that's the
context it's designed for. it deliberately uses no national emblem, no IMD or
ISRO logo, has its own name, and carries a prototype notice in the masthead.
