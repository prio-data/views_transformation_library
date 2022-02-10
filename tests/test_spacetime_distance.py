import pandas as pd
import numpy as np
from unittest import TestCase
import sys

sys.path.insert(1,'/Users/jim/Work/ViEWS/ViEWS3/views-transformation-library/views_transformation_library')
sys.path.insert(1,'/Users/jim/Work/ViEWS/ViEWS3/views-transformation-library')


from views_transformation_library import spacetime_distance

class TestSpacetimeDistance(TestCase):
    """
    Tests for the spacetime distance transform
    """

    def test_(self):
    
        index_times=[100,101,102]
        index_pgcells=[0,1,2,720,721,722,1440,1441,1442]
        ged_sb_sample=[
                       0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   # time=100
                       0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,   # time=101
                       0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0  # time=102
                       ]
    
        index_sample=pd.MultiIndex.from_product([index_times, index_pgcells],names=['month', 'pgid'])
    
        df = pd.DataFrame(data=ged_sb_sample,index=index_sample,columns=['ged_sb',])

        return_values='distances'
        k=1
        nu=1.0
        power=0.0

        df_sptime_dist=spacetime_distance.get_spacetime_distances(df,return_values,k,nu,power)
        
        self.assertTrue(df_sptime_dist.loc[100,721].values[0]==999.0)
        
        self.assertTrue(df_sptime_dist.loc[101,721].values[0]==0.0)
        
        self.assertTrue(df_sptime_dist.loc[101,722].values[0]==0.5)
        
        self.assertTrue(np.isclose(df_sptime_dist.loc[101,2].values[0],1./np.sqrt(2),rtol=0.001))
        
        self.assertTrue(df_sptime_dist.loc[102,721].values[0]==1.0)
        
        self.assertTrue(np.isclose(df_sptime_dist.loc[102,720].values[0],np.sqrt(1.+0.5**2.),rtol=0.001))
        
        return_values='distances'
        k=1
        nu=10.0
        power=0.0
        
        df_sptime_dist=spacetime_distance.get_spacetime_distances(df,return_values,k,nu,power)

        self.assertTrue(df_sptime_dist.loc[100,721].values[0]==999.0)
        
        self.assertTrue(df_sptime_dist.loc[101,721].values[0]==0.0)
        
        self.assertTrue(df_sptime_dist.loc[101,722].values[0]==0.5)
        
        self.assertTrue(df_sptime_dist.loc[102,721].values[0]==10.0)
        
        return_values='weights'
        k=1
        nu=1.0
        power=2.0
        
        df_sptime_dist=spacetime_distance.get_spacetime_distances(df,return_values,k,nu,power)
        
        self.assertTrue(df_sptime_dist.loc[100,721].values[0]==999.0)
        
        self.assertTrue(df_sptime_dist.loc[101,721].values[0]==1.0)
        
        self.assertTrue(np.isclose(df_sptime_dist.loc[101,722].values[0],0.5**-power,rtol=0.001))
        
        return_values='weights'
        k=1
        nu=10.0
        power=2.0
        
        df_sptime_dist=spacetime_distance.get_spacetime_distances(df,return_values,k,nu,power)
        
        self.assertTrue(df_sptime_dist.loc[100,721].values[0]==999.0)
        
        self.assertTrue(df_sptime_dist.loc[102,721].values[0]==0.01)
        
        self.assertTrue(np.isclose(df_sptime_dist.loc[102,722].values[0],np.sqrt(0.5**2.+10.0**2.)**-power,rtol=0.001))
        
        ged_sb_sample=[
                       0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,   # time=100
                       0.0,0.0,0.0,0.0,3.0,0.0,0.0,0.0,0.0,   # time=101
                       0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0  # time=102
                       ]
    
        index_sample=pd.MultiIndex.from_product([index_times, index_pgcells],names=['month', 'pgid'])
    
        df = pd.DataFrame(data=ged_sb_sample,index=index_sample,columns=['ged_sb',])
        
        return_values='weights'
        k=1
        nu=10.0
        power=2.0
        
        df_sptime_dist=spacetime_distance.get_spacetime_distances(df,return_values,k,nu,power)
        
        self.assertTrue(np.isclose(df_sptime_dist.loc[102,722].values[0],3.*np.sqrt(0.5**2.+10.0**2.)**-power,rtol=0.001))