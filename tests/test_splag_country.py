import pandas as pd
import numpy as np
from unittest import TestCase
from views_transformation_library import splag_country

class TestSplagCountry(TestCase):
    """
    Tests for the country-level spatial lag
    """

    def test_(self):
    
        index_time=[229]
        index_country=[57,59,120,235,237,242]
        ged_sb_sample=[0.0,281.0,0.0,18.0,18.0,0.0]
    
        index_sample=pd.MultiIndex.from_product([index_time, index_country],names=['month', 'country'])
    
        
    
        df = pd.DataFrame(data=ged_sb_sample,index=index_sample,columns=['ged_sb',])

        kernel_inner=0
        kernel_width=0
        kernel_power=0
        norm_kernel=0

        df_splag=splag_country.get_splag_country(df,kernel_inner,kernel_width,kernel_power,norm_kernel)
        
        self.assertTrue(np.all(df_splag.values==0.0))

        kernel_inner=0
        kernel_width=1
        kernel_power=0
        norm_kernel=0

        df_splag=splag_country.get_splag_country(df,kernel_inner,kernel_width,kernel_power,norm_kernel)
        
        self.assertTrue(np.all(df.values==df_splag.values))
        
        kernel_inner=1
        kernel_width=1
        kernel_power=0
        norm_kernel=0

        df_splag=splag_country.get_splag_country(df,kernel_inner,kernel_width,kernel_power,norm_kernel)
        
        self.assertTrue(df_splag.loc[229,237].values[0]==299.0)