import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from views_transformation_library import utilities


def get_spacetime_distances(df,return_values='distances',k=1,nu=1.0,power=0.0):
    '''
    
    get_spacetime_distances
    
    For every point in the supplied df, uses scipy.spatial cKDTree to find the nearest k
    past events (where an event is any non-zero value in the input df) and returns either 
    the mean spacetime distance to the k events, or the mean of 
    (size of event)/(spacetime distance)**power.
    
    Arguments:
    
    df:            input df
    
    return_values: choice of what to return. Allowed values:
     
                   distances - return mean spacetime distance to nearest k events     
    
                   weights   - return mean of (size of event)/(spacetime distance)**power
                   
    k:             number of nearest events to be averaged over
    
    nu:            weighting to be applied to time-component of distances, so that 
                   spacetime distance = sqrt(delta_latitude^2 + delta_longitude^2 +
                   nu^2*delta_t^2)
                   
    power:         power to which distance is raised when computing weights. Negative
                   values are automatically converted to positive values.
    
    '''

    if return_values not in ['distances','weights']:
        raise Exception("unknown return_values; ",return_values,
                        " - allowed choices: distances weights")
                        
    power=np.abs(power)

    times,time_to_index,index_to_time=utilities._map_times(df)

    pgids,pgid_to_longlat,longlat_to_pgid,pgid_to_index,index_to_pgid,ncells,pwr=\
                                                           utilities._map_pgids_2d(df)

    features=utilities._map_features(df)

    use_stride_tricks=True

    tensor=utilities._build_4d_tensor(
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

    df_stdist=space_time_distances(
                                    df,
                                    tensor,
                                    times,
                                    pgids,
                                    features,
                                    longlat_to_pgid,
                                    time_to_index,
                                    ncells,
                                    return_values,
                                    k,
                                    nu,
                                    power
                                    )
                                    
    return df_stdist

def space_time_distances(
                          df,
                          tensor,
                          times,
                          pgids,
                          features,
                          longlat_to_pgid,
                          time_to_index,
                          ncells,
                          return_values,
                          k,
                          nu,
                          power
                          ):
                          
    '''
    
    space_time_distances
    
    Finds spacetime distances, scaling spatial distances to degrees (for continuity with
    previous versions of this function) and stretching the time axis by nu.
    
    If k>1, averaging is performed.
    
    '''                      

    PGID_TO_DEGREES=0.5

    final=np.empty((len(times)*len(pgids),len(features)))

    npg=len(pgids)
    pgids_for_index=[]


    for ilong in range(ncells):
        for ilat in range(ncells):
            if (ilong,ilat) in longlat_to_pgid.keys():
                pgid=longlat_to_pgid[(ilong,ilat)]
                pgids_for_index.append(pgid)

    for time in times[0:600]:
    
        tindex=time_to_index[time]
        ipgid=0

        points=np.array(np.where(tensor[:,:,:tindex+1,0]>0)).astype(float).T
        

        if len(points)==0:
            pass
        else: 
            points[:,0]*=PGID_TO_DEGREES
            points[:,1]*=PGID_TO_DEGREES
            points[:,2]*=nu
            btree = cKDTree(data=points,leafsize=20)
                
        for ilong in range(ncells):
            for ilat in range(ncells):

                try:   
   
                    pgid=longlat_to_pgid[(ilong,ilat)]
                  
                    if len(points)==0:

                        feature=999.
 
                    else:

                        sptime_dists,ipoints=btree.query([ilong*PGID_TO_DEGREES,ilat*PGID_TO_DEGREES,nu*tindex],k)

                        if k==1:
                            ipoints=[ipoints,]
                            sptime_dists=[sptime_dists,]
                                
                        if return_values=='distances':
                                
                            if k==1:
                                feature=sptime_dists[0]
                            else:   
                                feature=np.mean(sptime_dists)
                        
                        elif return_values=='weights':

                            featurei=np.zeros(k)
                           
                            for i,ipoint in enumerate(ipoints):

                                coords=points[ipoint]
                                
                                ilongp=int(coords[0]/PGID_TO_DEGREES)
                                ilatp=int(coords[1]/PGID_TO_DEGREES)
                                itimep=int(coords[2]/nu)

                                if sptime_dists[i]==0.0:
                                    featurei[i]=tensor[ilongp,ilatp,itimep,0]
                                else:
                                    featurei[i]=tensor[ilongp,ilatp,itimep,0]/(sptime_dists[i]**(power))
                        
                            feature=np.mean(featurei)
                
                    indx=tindex*npg+ipgid
                    
                    final[indx,0]=feature

                    ipgid+=1
                except:
                    pass

    index_names=df.index.names
        
    df_index=pd.MultiIndex.from_product([times, pgids_for_index],names=index_names) 
    
    if return_values=='distances':
        colnames=['st_distances_nu_'+str(nu)+'_'+feature for feature in features]
    else:
        colnames=['st_weights_nu_'+str(nu)+'_k_'+str(k)+'_pwr_'+str(power)+'_'+feature for feature in features]
       
    df_stdist=pd.DataFrame(data=final,columns=colnames,index=df_index)

    df_stdist=df_stdist.sort_index()

    return df_stdist
