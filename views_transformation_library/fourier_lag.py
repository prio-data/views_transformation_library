import numpy as np
import scipy 
from scipy.fftpack import fft2,ifft2,dst,idst
import pandas as pd

def get_fourier_lag(df,dimensionality,use_stride_tricks=True):

    '''
    get_fourier_lag
    
    Driver function for obtaining spatial lags by using Fourier transforms to perform a 
    fast solution of Poisson's equation
    
    Arguments:
    
    df: dataframe containing one or more features to be lagged
    
    dimensionality: dimensionality of Fourier transforms to be used, which controls the 
    distance-weighting of the spatial lag. This can be 
    
    - 2, resulting in a lagging proportional to -ln(d)
    - 3, resulting in a lagging proportional to 1/d
    
    use_stride_tricks: boolean, controlling whether stride tricks is used for tensor
    transforms
    
    '''

    if dimensionality==2:
        transformer=fft_2D
    
    elif dimensionality==3:
        transformer=fft_3D
    
    else:
         print('Dimensionality ',dimensionality,' not supported.')
         return None
         
    pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,index_to_pgid,longrange,latrange=map_pgids(df)     
    
    times,time_to_index,index_to_time=map_times(df)
    
    features=map_features(df)
    
    tensor4d=build_4d_tensor(df,pgids,pgid_to_index,times,time_to_index,longrange,latrange,pgid_to_longlat,features,use_stride_tricks)
    
    flags=transformer(tensor4d,times,features,time_to_index)
    
    df_flags=flags_to_df(flags,times,time_to_index,pgids,features,longrange,latrange,longlat_to_pgid)
    
    return df_flags

def map_pgids(df):

    '''
	map_pgids
	   
	This function builds a 2D map in longitude-latitude from the pgids contained in 
	the input dataframe, and creates dicts allowing quick transformation from (long,lat)
	to pgid and vice versa.
	   
	The pgids are embedded and centred in the smallest possible square grid whose side
	is an integer power of 2
	'''
	
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
        
    latmin=np.min(latitudes)
    latmax=np.max(latitudes)
    longmin=np.min(longitudes)
    longmax=np.max(longitudes)
       
    maxsize=np.max((longrange,latrange))
    power=1+int(np.log2(maxsize))

    ncells=2**power
        
    inudgelong=int((ncells-longmax)/2)
    inudgelat=int((ncells-latmax)/2)

    longitudes+=inudgelong
    latitudes+=inudgelat
    
    latmin=np.min(latitudes)
    latmax=np.max(latitudes)
    longmin=np.min(longitudes)
    longmax=np.max(longitudes)

    latrange=ncells
    longrange=ncells

	# make dicts to transform between pgids and (long,lat) coordinate      

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
    
def map_times(df):

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

    tensor4d=np.empty((longrange,latrange,len(times),len(features)))

    for pgid in pgids:

        pgindex=pgid_to_index[pgid]
        for time in times:
        
            tindex=time_to_index[time]
            ilong=pgid_to_longlat[pgid][0]
            ilat=pgid_to_longlat[pgid][1]
            tensor4d[ilong,ilat,tindex,:]=tensor3d[tindex,pgindex,:]
            
    return tensor4d
    
    
def fft_2D(tensor4d,times,features,time_to_index):

    '''
    fft_2D
    
    Function which uses a 2D Fourier transform to solve Poisson's equation in 2 spatial
    dimensions.
    
    This results in distance weighting of the lagged variable of -ln(d)
    
    
    '''

    # small value to use instead of zero for zero spatial frequencies

    SMALL=1e-3

    dim0=np.shape(tensor4d)[0]
    transformed=np.empty_like(tensor4d)
    
    freqsj=freqsk=None
    rangej=rangek=dim0
    
    for ifeature,feature in enumerate(features):
        for time in times:

            itime=time_to_index[time]
            
            rhomax=np.max(tensor4d[:,:,itime,ifeature])

            # perform 2D FFT on time slice from tensor

            rhojkhat=np.fft.fft2(tensor4d[:,:,itime,ifeature],norm='ortho')

            if freqsj is None:

               # get spatial frequencies

               freqsj=2.*np.pi*np.fft.fftfreq(rhojkhat.shape[0])
               freqsk=2.*np.pi*np.fft.fftfreq(rhojkhat.shape[1])
       
               freqsj[abs(freqsj)<SMALL]=SMALL
               freqsk[abs(freqsk)<SMALL]=SMALL

               # make tensor with values 1/(j**2+k**2)

               divisor=np.ones((rangej,rangek))
       
               for ij in range(rangej):
                   ifreqj=freqsj[ij]
                   ifreqj2=ifreqj*ifreqj
                   for ik in range(rangek):
                       ifreqk=freqsk[ik]
                       ifreqk2=ifreqk*ifreqk
                       divisor[ij,ik]/=(ifreqj2+ifreqk2)

            # generate solution to Poisson's equation on slice

            phijkhat=np.multiply(rhojkhat,divisor)
            
            # take real part of inverse transform
            
            phijk=np.real(np.fft.ifft2(phijkhat,norm='ortho'))
            
            # subtract zero level
            
            phijk-=np.min(phijk)
            
            # rescale lagged variable to have same scale as original feature
            
            phimax=np.max(phijk)

            phimax=np.max([phimax,1.0])
            
            transformed[:,:,itime,ifeature]=phijk*rhomax/phimax

    return transformed


def fft_3D(tensor4d,times,features,time_to_index):

    '''
    fft_3D
    
    Function which uses a 3D Fourier transform to solve Poisson's equation in 3 spatial
    dimensions.
    
    This results in distance weighting of the lagged variable of 1/d
    
    This is done by embedding the flat array from the original feature as the midplane of 
    a cube, so that the problem effectively becomes 3-dimensional.
    
    To save memory and time, the number of Fourier modes used in the extra dimension is 
    reduced by a factor LDIVISOR. Experimentation shows that adequate resolution can be
    achieved with LDIVISOR set to 4
    
    
    '''

    # factor by which to reduce number of Fourier modes in extra dimension

    LDIVISOR=4
    
    # small value to use instead of zero for zero spatial frequencies
    
    SMALL=1e-3
    
    dim0=np.shape(tensor4d)[0]
    
    transformed=np.empty_like(tensor4d)

    freqsj=freqsk=freqsl=None
    rangej=rangek=dim0
    rangel=int(dim0/LDIVISOR)
    
    for ifeature,feature in enumerate(features):
        for time in times:
          
            itime=time_to_index[time]
            
            rhomax=np.max(tensor4d[:,:,itime,ifeature])
            
            # embed time slice from tensor into a 3D tensor

            rhojkl=np.zeros((rangej,rangek,rangel))
        
            icentrej=icentrek=int(rangej/2)
            icentrel=int(rangel/2)
        
            rhojkl[:,:,icentrel]=tensor4d[:,:,itime,ifeature]
            
            # perform N-dimensional FFT on embedded time slice

            rhojklhat=np.fft.fftn(rhojkl,norm='ortho')

            if freqsj is None:

               # get spatial frequencies

               freqsj=2.*np.pi*np.fft.fftfreq(rhojklhat.shape[0])
               freqsk=2.*np.pi*np.fft.fftfreq(rhojklhat.shape[1])
               freqsl=2.*np.pi*np.fft.fftfreq(rhojklhat.shape[2])
       
               freqsj[abs(freqsj)<SMALL]=SMALL
               freqsk[abs(freqsk)<SMALL]=SMALL
               freqsl[abs(freqsl)<SMALL]=SMALL

               # make tensor with values 1/(j**2+k**2+l**2)

               divisor=np.ones((rangej,rangek,rangel))
       
               for ij in range(rangej):
                   ifreqj=freqsj[ij]
                   ifreqj2=ifreqj*ifreqj
                   for ik in range(rangek):
                       ifreqk=freqsk[ik]
                       ifreqk2=ifreqk*ifreqk
                       for il in range(rangel):
                           ifreql=freqsl[il]
                           ifreql2=ifreql*ifreql
            
                           divisor[ij,ik,il]/=(ifreqj2+ifreqk2+ifreql2)

            # generate solution to Poisson's equation on embedded slice

            phijklhat=np.multiply(rhojklhat,divisor)
            
            # take real part of inverse transform
            
            phijkl=np.real(np.fft.ifftn(phijklhat,norm='ortho'))
            
            # subtract zero level
            
            phijkl-=np.min(phijkl)
            
            # rescale lagged variable to have same scale as original feature
            
            phimax=np.max(phijkl)

            phimax=np.max([phimax,1.0])

            transformed[:,:,itime,ifeature]=phijkl[:,:,icentrel]*rhomax/phimax
        
    return transformed

def flags_to_df(
    flags,
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
                    final[indx,:]=flags[ilong,ilat,tindex,:]

                ipgid+=1
            except:
                pass
            
    splags_index=pd.MultiIndex.from_product([list(times),pgids_for_index])
    colnames=['splag_'+feature for feature in features]
    df_flags=pd.DataFrame(data=final,columns=colnames,index=splags_index)

    df_flags=df_flags.sort_index()

    return df_flags
    


