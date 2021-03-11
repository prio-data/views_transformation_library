
# ViEWS Transformation Library

This package contains transforms made available in ViEWS cloud.  The transforms
are all locally available, and do not depend on remote resources or
credentials.

## Install:

```
pip install views-transformation-library
```

## Usage:

```
from views_transformation_library.timelag import timelag
import viewser

ged_best_ns = viewser.get_variable("priogrid_month","ged_best_ns",2010,2020)
lagged_data = timelag(ged_best_ns,10)

lagged_from_api = viewser.get_variable("priogrid_month","ged_best_ns",2010,2020,transforms:[
   {"type":"tlag","args":[10]}
])

assert lagged_from_api == lagged_data
```
