import numpy as np
import pandas as pd

from scipy import ndimage

from views_transformation_library import utilities
   
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
   
    kernel_inner:      inner border of convolution region (set to 1 to exclude central
                       cell)

    kernel_width:      width in cells of kernel, so outer radius of kernel =
                       kernel_inner + kernel_width

    kernel_power:      weight values of cells by (distance from centre of kernel)**
                       (-kernel_power) - set to zero for no distance weighting

    norm_kernel:       set to 1 to normalise kernel weights
                       
    Returns:
    
    A df of all lagged columns from input df
    
    '''

    arg_string=str(kernel_inner)+'_'+str(kernel_width)+'_'+str(kernel_power)+'_'+str(norm_kernel)

    df=df.fillna(0.0)
    if not(df.index.is_monotonic):
        df=df.sort_index()
 
    weights=build_kernel_weights(kernel_inner,kernel_width,kernel_power,norm_kernel)
     
    times,time_to_index,index_to_time=utilities._map_times(df)

    features=utilities._map_features(df)

    pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,index_to_pgid,ncells,power=\
                                                          utilities._map_pgids_2d(df)
    
    tensor4d=utilities._build_4d_tensor(
                                       df,
                                       pgids,
                                       pgid_to_index,
                                       times,
                                       time_to_index,
                                       ncells,
                                       ncells,
                                       pgid_to_longlat,
                                       features,
                                       use_stride_tricks
                                       )

    splags=get_splags(tensor4d,ncells,ncells,times,features,time_to_index,weights)

    df_splags=splags_to_df(
                           df,
                           splags,
                           times,
                           time_to_index,
                           pgids,
                           features,
                           ncells,
                           ncells,
                           longlat_to_pgid,
                           arg_string
                           )

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
    
def get_splags(
    tensor4d,
    longrange,
    latrange,
    times,
    features,
    time_to_index,
    weights
):

    splags=np.empty((longrange,latrange,len(times),len(features)))
    
    # use scipy convolution function to do 2d convolution on tensor slices
    
    for time in times:
        tindex=time_to_index[time]
        
        for ifeature in range(len(features)):
            raw=tensor4d[:,:,tindex,ifeature]
            splags[:,:,tindex,ifeature]=ndimage.convolve(raw, weights, mode='constant', cval=0.0)

    return splags
    
def splags_to_df(df,
    splags,
    times,
    time_to_index,
    pgids,
    features,
    longrange,
    latrange,
    longlat_to_pgid,
    argstring
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
            
    index_names=df.index.names
        
    df_index=pd.MultiIndex.from_product([list(times),pgids_for_index],names=index_names)
        
    colnames=['splag_'+argstring+'_'+feature for feature in features]
    df_splags=pd.DataFrame(data=final,columns=colnames,index=df_index)

    df_splags=df_splags.sort_index()

    return df_splags
    
    

