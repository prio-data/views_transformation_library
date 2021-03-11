
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
