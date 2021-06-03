import numpy as np
import pandas as pd

from scipy import ndimage
   
def get_splag4d(
    df,
    use_stride_tricks=True,
    kernel_inner=1,kernel_width=1,kernel_power=0,norm_kernel=0

):

    '''
    splag4d created 19/03/2021 by Jim Dale
    
    Performs spatial lags on a dataframe by transforming from flat format to 4d tensor 
    with dimensions longitude x latitude x time x features.
    
    Spatial lagging can then be done as a 2d convolution on long-lat slices using
    scipy convolution algorithms.
    
    Arguments:
    
    df:                a dataframe of series to be splagged
    
    use_stride_tricks: boolean, decide to use stride_tricks or not (optional, 
                       defaults to True)
                       
    weights:           weight matrix to use in convolutions (optional, defaults to
                       first-order lag, with zero at [1,1])
                       
    Returns:
    
    A df of all lagged columns from input df
    
    '''

    df=df.fillna(0.0)
    if not(df.index.is_monotonic):
        df=df.sort_index()
 
    weights=build_kernel_weights(kernel_inner,kernel_width,kernel_power,norm_kernel)
     
    times,time_to_index,index_to_time=map_times(df)

    features=map_features(df)

    pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,index_to_pgid,longrange,latrange=map_pgids(df)

    tensor4d=build_4d_tensor(df,pgids,pgid_to_index,times,time_to_index,longrange,latrange,pgid_to_longlat,features,use_stride_tricks)

    splags=get_splags(tensor4d,longrange,latrange,times,features,time_to_index,weights)

    df_splags=splags_to_df(splags,times,time_to_index,pgids,features,longrange,latrange,longlat_to_pgid)

    return df_splags

def build_kernel_weights(kernel_inner,kernel_width,kernel_power,norm_kernel):
    
    kernel_inner=int(kernel_inner)
    kernel_width=int(kernel_width)
    
    kernel_size=int(2*(kernel_inner+kernel_width))-1
    weights=np.ones((kernel_size,kernel_size))
    
    kernel_centre=int((kernel_size+1)/2)-1
    
    for ix in range(kernel_size):
        dx=ix-kernel_centre
        for iy in range(kernel_size):
            dy=iy-kernel_centre
            if ((abs(dx)<kernel_inner) and (abs(dy)<kernel_inner)):
               weights[ix,iy]=0.0
            else:
               rxy=np.sqrt(dx*dx+dy*dy)
               weights[ix,iy]/=rxy**kernel_power
               
    if (norm_kernel):
       weights/=np.sum(weights)
       
    return weights
    

def df_to_tensor_strides(
    df
):
    '''
    df_to_tensor created 13/03/2021 by Jim Dale
    Uses as_strided from numpy stride_tricks library to create a tensorlike
    from a dataframe.
    
    There are two possibilities:
    
    (i) If the dataframe is uniformly-typed, the tensor is a memory view: 
    changes to values in the tensor are reflected in the source dataframe.

    (ii) Otherwise, the tensor is a copy of the dataframe and changes in it
    are not propagated to the dataframe.

    Note that dim0 of the tensor corresponds to level 0 of the df multiindex,
    dim1 corresponds to level 1 of the df multiindex, and dim2 corresponds to 
    the df's columns.
    '''

    # get shape of dataframe

    dim0,dim1=df.index.levshape
    
    dim2=df.shape[1]
    
    # check that df can in principle be tensorised
    
    if dim0*dim1!=df.shape[0]:
        raise Exception("df cannot be cast to a tensor - dim0 * dim1 != df.shape[0]",dim0,dim1,df.shape[0])

    flat=df.to_numpy()

    # get strides (in bytes) of flat array
    flat_strides=flat.strides

    offset2=flat_strides[1]

    offset1=flat_strides[0]
    
    # compute stride in bytes along dimension 0
    offset0=dim1*offset1

    # get memory view or copy as a numpy array
    tensor3d=np.lib.stride_tricks.as_strided(flat,shape=(dim0,dim1,dim2),
                                           strides=(offset0,offset1,offset2))
                                           
    return tensor3d

def df_to_tensor_no_strides(
    df
):
    
    # get shape of dataframe
    
    dim0,dim1=df.index.levshape
    
    dim2=df.shape[1]
    
    # check that df can in principle be tensorised
    
    if dim0*dim1!=df.shape[0]:
        raise Exception("df cannot be cast to a tensor - dim0 * dim1 != df.shape[0]",dim0,dim1,df.shape[0])

    
    flat=df.to_numpy()

    tensor3d=np.empty((dim0,dim1,dim2))
    chunksize=dim1
    nchunks=dim0
    for ichunk in range(nchunks):
        tensor3d[ichunk,:,:]=flat[ichunk*chunksize:(ichunk+1)*chunksize,:]

    return tensor3d
    
def map_times(
    df
):

    # get unique times

    times=np.array(list({idx[0] for idx in df.index.values}))
    times=list(np.sort(times))
    
    # make dicts to transform between times and the index of a time in the list
    
    time_to_index={}
    index_to_time={}
    for i,time in enumerate(times):
        time_to_index[time]=i
        index_to_time[i]=time
    
    return times,time_to_index,index_to_time
    
def map_features(
    df
):

    features=list(df.columns)
    
    return features
    
def map_pgids(
    df
):
    
    PG_STRIDE=720
    
    # get unique pgids
    
    pgids=np.array(list({idx[1] for idx in df.index.values}))
    pgids=np.sort(pgids)
        
    # convert pgids to longitudes and latitudes
    
    longitudes=pgids%PG_STRIDE
    latitudes=pgids//PG_STRIDE
    
    latmin=np.min(latitudes)
    latmax=np.max(latitudes)
    longmin=np.min(longitudes)
    longmax=np.max(longitudes)

    latrange=latmax-latmin
    longrange=longmax-longmin

    # shift to a set of indices that starts at [0,0]

    latitudes-=latmin
    longitudes-=longmin

    # add padding of one cell 

    latitudes+=1
    longitudes+=1

    # make dicts to transform between pgids and (long,lat) coordinates

    pgid_to_longlat={}
    longlat_to_pgid={}
    
    pgid_to_index={}
    index_to_pgid={}

    for i,pgid in enumerate(pgids):
        pgid_to_longlat[pgid]=(longitudes[i],latitudes[i])
        longlat_to_pgid[(longitudes[i],latitudes[i])]=pgid
        pgid_to_index[pgid]=i
        index_to_pgid[i]=pgid

    return pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,index_to_pgid,longrange,latrange
    
def build_4d_tensor(
    df,
    pgids,
    pgid_to_index,
    times,
    time_to_index,
    longrange,
    latrange,
    pgid_to_longlat,
    features,
    use_stride_tricks
):
    
    # convert flat data from df into time x pgid x feature tensor
    
    if(use_stride_tricks):
        tensor3d=df_to_tensor_strides(df)
    else:
        tensor3d=df_to_tensor_no_strides(df)

    # convert 3d tensor into longitude x latitude x time x feature tensor

    tensor4d=np.empty((longrange+2,latrange+2,len(times),len(features)))

    for pgid in pgids:

        pgindex=pgid_to_index[pgid]
        for time in times:
        
            tindex=time_to_index[time]
            ilong=pgid_to_longlat[pgid][0]
            ilat=pgid_to_longlat[pgid][1]
            tensor4d[ilong,ilat,tindex,:]=tensor3d[tindex,pgindex,:]
            
    return tensor4d
    
def get_splags(
    tensor4d,
    longrange,
    latrange,
    times,
    features,
    time_to_index,
    weights
):

    splags=np.empty((longrange+2,latrange+2,len(times),len(features)))
    
    # use scipy convolution function to do 2d convolution on tensor slices
    
    for time in times:
        tindex=time_to_index[time]
        
        for ifeature in range(len(features)):
            raw=tensor4d[:,:,tindex,ifeature]
            splags[:,:,tindex,ifeature]=ndimage.convolve(raw, weights, mode='constant', cval=0.0)

    return splags
    
def splags_to_df(
    splags,
    times,
    time_to_index,
    pgids,
    features,
    longrange,
    latrange,
    longlat_to_pgid
):

    final=np.empty((len(times)*len(pgids),len(features)))

    # convert 4d spatial lag tensor to flat table

    ipgid=0
    npg=len(pgids)
    pgids_for_index=[]

    for ilong in range(1,longrange+2):
        for ilat in range(1,latrange+2):
            try:
                pgid=longlat_to_pgid[(ilong,ilat)]
                pgids_for_index.append(pgid)
                for time in times:
                    tindex=time_to_index[time]
                    indx=tindex*npg+ipgid
                    final[indx,:]=splags[ilong,ilat,tindex,:]

                ipgid+=1
            except:
                pass
            
    splags_index=pd.MultiIndex.from_product([list(times),pgids_for_index])
    colnames=['splag_'+feature for feature in features]
    df_splags=pd.DataFrame(data=final,columns=colnames,index=splags_index)

    df_splags=df_splags.sort_index()

    return df_splags
    
    

