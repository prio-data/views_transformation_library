
import unittest

import numpy as np
import pandas as pd

from views_transformation_library.missing_handling import replace_na

class TestMissingHandling(unittest.TestCase):
    def test_simple_replacement(self):
        dataframe = (pd.DataFrame(np.ones((10,10)))
                .shift(5)
                )

        dataframe = replace_na(dataframe,0)

        self.assertEqual(
                dataframe.isna().sum().sum(),0
            )
        self.assertEqual(dataframe.sum().sum(),50)
