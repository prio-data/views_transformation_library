
# ViEWS Transformation Library

This package contains transforms made available in ViEWS cloud.  The transforms
are all locally available, and do not depend on remote resources or
credentials.

## Contributing: 

There is an urgent need to fill this library with transforms.
Transforms we need, that are not implemented, are listed under issues.
Please follow the guidelines for contributing outlined in the [CONTRIBUTING.md](CONTRIBUTING.md) document.

## Usage:

```
pip install views-transformation-library
```

```
from views_transformation_library.timelag import timelag
import viewser

ged_best_ns = viewser.get_variable("priogrid_month","priogrid_month.ged_best_ns",2010,2020)
lagged_data = timelag(ged_best_ns,10)

lagged_from_api = viewser.get_variable("priogrid_month","ged_best_ns",2010,2020,transforms:[
   {"type":"tlag","args":[10]}
])

assert lagged_from_api == lagged_data
```

## API:

Each transform function operates on a pandas dataframe.

## Funding

The contents of this repository is the outcome of projects that have received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 694640, *ViEWS*) and Horizon Europe (Grant agreement No. 101055176, *ANTICIPATE*; and No. 101069312, *ViEWS* (ERC-2022-POC1)), Riksbankens Jubileumsfond (Grant agreement No. M21-0002, *Societies at Risk*), Uppsala University, Peace Research Institute Oslo, the United Nations Economic and Social Commission for Western Asia (*ViEWS-ESCWA*), the United Kingdom Foreign, Commonwealth & Development Office (GSRA – *Forecasting Fatalities in Armed Conflict*), the Swedish Research Council (*DEMSCORE*), the Swedish Foundation for Strategic Environmental Research (*MISTRA Geopolitics*), the Norwegian MFA (*Conflict Trends* QZA-18/0227), and the United Nations High Commissioner for Refugees (*the Sahel Predictive Analytics project*).
