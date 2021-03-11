import unittest
import pandas as pd
from views_transformation_library.identity import identity

class TestIdentity(unittest.TestCase):
    def test_identity(self):
        data = pd.DataFrame([{"a":1,"b":2},{"a":2,"b":1}])
        self.assertTrue(identity(data).equals(data))
