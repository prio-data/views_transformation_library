import numpy as np
import scipy 
from scipy.fftpack import fft2,ifft2,dst,idst
import pandas as pd
from views_transformation_library import utilities

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
         
    pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,\
    index_to_pgid,ncells,power=utilities._map_pgids_2d(df)     
    
    times,time_to_index,index_to_time=utilities._map_times(df)
    
    features=utilities._map_features(df)
    
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
    
    flags=transformer(tensor4d,times,features,time_to_index)
    
    df_flags=flags_to_df(
                         df,
                         flags,
                         times,
                         time_to_index,
                         pgids,
                         features,
                         ncells,
                         ncells,
                         longlat_to_pgid
                         )
    
    return df_flags
    
    
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
    df,
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
            
    index_names=df.index.names
        
    df_index=pd.MultiIndex.from_product([list(times),pgids_for_index],names=index_names)
            
    colnames=['flag_'+feature for feature in features]
    df_flags=pd.DataFrame(data=final,columns=colnames,index=df_index)

    df_flags=df_flags.sort_index()

    return df_flags
    


