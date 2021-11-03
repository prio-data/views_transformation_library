import numpy as np
import pandas as pd
from views_transformation_library import utilities

def tlags3d(
    df,
    use_stride_tricks=True,
    lags=[1,]
):

    '''
    tlag3d created 22/03/2021 by Jim Dale
    
    Performs temporal lags on a dataframe by transforming from flat format to 3d tensor 
    with dimensions time x pgid x features.
    
    Accepts arbitrary numbers of columns in input df, and arbitrary list of lags.
    
    Arguments:
    
    df:                a dataframe of series to be timelagged
    
    use_stride_tricks: boolean, decide to use stride_tricks or not (optional, 
                       defaults to True)
                       
    lags:              a list of lags to be applied (optional, defaults to [1,])
    
    Returns:
    
    A dataframe with all lags of all columns in input df
    
    '''


    df=df.fillna(0.0)
    df=df.sort_index()
    df_index=df.index

    times,time_to_index,index_to_time=utilities._map_times(df)

    features=utilities._map_features(df)

    pgids,pgid_to_index,index_to_pgid=utilities._map_pgids_1d(df)

    if (use_stride_tricks):
        tensor3d=utilities._df_to_tensor_strides(df)
    else:
        tensor3d=utilities._df_to_tensor_no_strides(df)

    tlags=get_tlags(times,pgids,features,time_to_index,tensor3d,lags)

    df_tlags=tlags_to_df_strides(tlags,times,time_to_index,pgids,features,lags,df_index)

    return df_tlags
    
def get_tlags(
    times,
    pgids,
    features,
    time_to_index,
    tensor3d,
    lags
):

    tlags=np.zeros((len(times),len(pgids),len(features)*len(lags)))

    for ifeature in range(len(features)):
        for ilag,lag in enumerate(lags):
            for time in times:
                tindex=time_to_index[time]            
                try:
                    tlags[tindex,:,ifeature*len(lags)+ilag]=tensor3d[tindex-lag,:,ifeature]
                except:
                    pass  

    return tlags
    
def tlags_to_df_no_strides(
    tensor3d,
    tlags,
    times,
    time_to_index,
    pgids,
    features,
    lags,
    df_index
):

    dim0,dim1,dim2=len(times),len(pgids),len(features)*len(lags)
    
    flat=np.empty((dim0*dim1,dim2))
    
    for ichunk in range(dim0):
        flat[ichunk*dim1:(ichunk+1)*dim1,:]=tensor3d[ichunk,:,:]
       
    df_column_names=['tlag_'+str(lag)+'_'+feature for feature in features for lag in lags]
        
    df_tlags=pd.DataFrame(flat, index=df_index, columns=df_column_names)
    
    return df_tlags    
    
    
def tlags_to_df_strides(
    tlags,
    times,
    time_to_index,
    pgids,
    features,
    lags,
    df_index
):

    dim0,dim1,dim2=len(times),len(pgids),len(features)*len(lags)
    
    flat=np.empty((dim0*dim1,dim2))
    
    tlags_strides=tlags.strides

    offset2=tlags_strides[2]

    offset1=tlags_strides[1]
    
    offset0=tlags_strides[0]
    
    flat=np.lib.stride_tricks.as_strided(tlags,shape=(dim0*dim1,dim2),
                                           strides=(offset1,offset2))
       
    df_column_names=['tlag_'+str(lag)+'_'+feature for feature in features for lag in lags]
        
    df_tlags=pd.DataFrame(flat, index=df_index, columns=df_column_names)
    
    return df_tlags    
    