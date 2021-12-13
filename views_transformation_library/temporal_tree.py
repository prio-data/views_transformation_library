import pandas as pd
import numpy as np
from views_transformation_library import utilities


def get_tree_lag(df,thetacrit,weight_functions,sigma,use_stride_tricks):
    
    '''
    get_tree_lag
    
    Driver function for computing temporal-tree-lagged features
    
    Arguments are:
    
    df: dataframe containing one or more features to be lagged
    thetacrit: parameter controlling how aggressively nodes in the past are aggregated
    weight_functions: list containing one or more of the following strings:
    
    - 'uniform': weights for all nodes are unity. Unlikely to be meaningful but provided for completeness
    - 'oneovert': weights for nodes are 1/(tnow-tnode)
    - 'expon': weights for nodes are exp(-(tnow-tnode)/sigma)
    - 'ramp': weights for nodes are 1-(tnow-tnode)/sigma for (tnow-tnode)<sigma, 0 otherwise
    - 'sigmoid': weights are 1./(1+np.exp(-lag)) where lag=(mid-tnow+5*sigma5)/sigma5 and sigma5=sigma/5
    
    sigma: parameter in time units used by df controlling width of expon, ramp and sigmoid functions
    
    '''
    
    df=df.fillna(0.0)
    if not(df.index.is_monotonic):
        df=df.sort_index()
        
    if type(weight_functions)!='list':
        weight_functions=[weight_functions,]
    
    tree=TemporalTree()

    tree.build_tree(df)
    
    tree.stock_initial(df,use_stride_tricks)
    
    df_treelags=tree.tree_lag(df,thetacrit,weight_functions,sigma)
    
    return df_treelags

class TemporalNode():
    """
    TemporalNode() class
    Defines nodes for insertion in TemporalTree() class
    
     Nodes have:
    - nodeid: a unique id
    - level: integer indicating which level of the tree the node belongs to, 0 is leaf level
    - parent: parent node
    - start, end: node boundaries
    - sibling: id of node's sibling
    - predecessor: id of immediately preceding node at same tree level
    - ispast: flag indicating whether node is past or future compared to its sibling 
    - isleaf: a flag indicating whether a leaf node
    - nleaf: the number of leaf nodes contained within the node's boundaries
    - children: list of node's children
    - features: dict of features loaded into the node by the .stock() method
    
    """    
    def __init__(self,nodeid,level,start,end,parent,sibling,predecessor,ispast,isleaf):
        self.nodeid = nodeid
        self.level = level
        self.start = start
        self.end = end
        self.parent = parent
        self.sibling = sibling
        self.predecessor = predecessor
        self.ispast = ispast
        self.isleaf = isleaf
        self.children = []
        self.nleaf = 0
        self.features = {}
        
    def __repr__(self):
        return f""" nodeid: {self.nodeid}\n level: {self.level}\n parent: {self.parent}\n start: {self.start}
        end: {self.end}\n bottom: {self.bottom}\n isleaf: {self.isleaf}\n children: {self.children}
        predecessor: {self.predecessor}\n sibling: {self.sibling}\n"""

class TemporalTree():
    """
    TemporalTree() class
    """  
    def __init__(self):
        self.gridsize=0
        self.nodes=[]
        self.maxlevel=-1
        self.stocked=False
        self.stocked_until=-1
        self.weight_fn=None
        self.timesequence=[]
        self.npad=0
        self.times=[]
        self.time_to_index={}
        self.index_to_time={}
        self.pgid_to_index={}
        self.index_to_pgid={}
        self.features = {}
        self.weight_functions={}
        
        self.make_weight_functions()
    
    def build_tree(self,df):
       
        '''
	    build_tree
	   
	    This function builds the tree from the leaf nodes downwards.
	   
	    The total number of leaf nodes must be an integer power of two, so the list of 
	    times is padded with dummy nodes placed before the earliest time the tree is
	    built on.
	   
	    Once leaf nodes are populated, work down through the levels. Assign parents,
	    siblings and predecessors
	   
	    '''
       
        self.times,self.time_to_index,self.index_to_time=utilities._map_times(df)
        
        tstart=self.times[0]
        tend=self.times[-1]
        
        nseq_initial=tend-tstart
        log2_nseq_initial=np.log2(nseq_initial)
        nseq=int(2**(1+int(log2_nseq_initial)))
        npad=nseq-nseq_initial
        
        self.timesequence=[t for t in range(tstart-npad,tend+1)]
        self.npad=npad
        
        nodestodo=[]
        nnodes=0
        level=0
        maxlevel=0
        parent=-1
        sibling=-1
        predecessor=-1
        ispast=False
        isleaf=False
        node=TemporalNode(nnodes,level,tstart-npad,tend,parent,sibling,predecessor,ispast,isleaf)
        self.nodes.append(node)
        nodestodo.append(node)
        
        while len(nodestodo)>0:
            splitnode=nodestodo.pop()
            
            if (splitnode.end-splitnode.start)>1:
                nnodes+=1
                if (splitnode.start>=0):
                    mid=int((splitnode.start+splitnode.end+1)/2)
                else:
                    mid=int((splitnode.start+splitnode.end)/2)

                level=splitnode.level+1
                maxlevel=max(maxlevel,level)
                parent=splitnode.nodeid
                sibling=None
                predecessor=None
                ispast=True
                if (mid-splitnode.start)>1:
                    isleaf=False
                else:
                    isleaf=True
                pastnode=TemporalNode(nnodes,level,splitnode.start,mid,parent,sibling,predecessor,ispast,isleaf)
                self.nodes.append(pastnode)
                nodestodo.append(pastnode)
                nnodes+=1
                ispast=False
                futurenode=TemporalNode(nnodes,level,mid,splitnode.end,parent,sibling,predecessor,ispast,isleaf)
                self.nodes.append(futurenode)
                nodestodo.append(futurenode)
                pastnode.sibling=futurenode.nodeid
                futurenode.sibling=pastnode.nodeid
                futurenode.predecessor=pastnode.nodeid
                splitnode.children=[pastnode.nodeid,futurenode.nodeid]
                
        self.maxlevel=maxlevel
        
        for node in self.nodes:
            while node.predecessor is None:
                if node.start<=(self.timesequence[0]+self.npad):
                    node.predecessor=-1
                else:
                    level=node.level
                    climb=True
                    climbnode=self.nodes[node.parent]
                    while climb:
                        if climbnode.ispast:
                            climbnode=self.nodes[climbnode.parent]
                        else:
                            descendnode=self.nodes[climbnode.sibling]
                            climb=False
                    descendlevel=descendnode.level

                    while descendnode.level!=level:
                        descendnode=self.nodes[descendnode.children[1]]
                    node.predecessor=descendnode.nodeid
        
    def stock_initial(self,df,use_stride_tricks):
    
        '''
        stock_initial
        
        This function loads features into the tree.
        
        Data from the input df is first cast to a 3D tensor with dims time x pgid x
        features.
        
        The tensor is used to populate the leaf nodes, and values then just propagate
        down through the tree via the parents.
        
        At present, values of parents are the *sums* of those of their children.
        
        '''
    
        if self.stocked:
            print('Tree has already been stocked - aborting')
            raise(Exception)
            
        self.stocked=True

        self.features=utilities._map_features(df)
        
        self.pgids,self.pgid_to_index,self.index_to_pgid=utilities._map_pgids_1d(df)
		
        npgids=len(self.pgids)
		
        for node in self.nodes:
            for feature in self.features:
                node.features[feature]=np.zeros(npgids)
                
        if(use_stride_tricks):
            tensor3d=utilities._df_to_tensor_strides(df)
        else:
            tensor3d=utilities._df_to_tensor_strides(df)

        
        for time in self.times:
            itime=self.time_to_index[time]
            for node in self.nodes:
                if (node.isleaf and node.start==time):
                    for ifeature,feature in enumerate(self.features):
                        vals=tensor3d[itime,:,ifeature]
                        node.features[feature]+=vals
                        node.nleaf=1
                        parentid=node.parent
                        while parentid!=-1:
                            parent=self.nodes[parentid]
                            parent.features[feature]+=vals
                            parent.nleaf+=1
                            parentid=parent.parent
                
        self.stocked_until=self.times[-1]
        
    def walk(self,tnow,thetacrit):
        
        '''
	    walk
	   
	    This function generates the list of nodes any given node will import data from, 
	    when one lags variables using the tree, as well as weights based on the times
	    between nodes' midpoints.
	    
	    The arguments are 
	    
	    - thetacrit: angle used to decide if a candidate node should be added to a target 
	      node's interaction list, based on the size of the candidate node and
	      the time gap between the candidate node and the target node 
	   
	    '''
        
        if ((tnow<self.nodes[0].start) or (tnow>self.nodes[0].end)):
            print('tnow not in range of times covered by this tree - aborting')
            raise(Exception)
        if (tnow>self.stocked_until):
            print('tree has not been stocked as far as tnow - aborting')
            raise(Exception)
            
        list_of_nodes=[]

        for node in self.nodes:
            if (node.isleaf and node.start==tnow):
                list_of_nodes.append(node.nodeid)
                if node.predecessor==-1:return list_of_nodes
                notdone=True
                while notdone:
                    if node.ispast:
                        if (node.predecessor==-1):
                            notdone=False
                        else:
                            pred=self.nodes[node.predecessor]
                            node=self.nodes[pred.parent]
                            self.split_node(node,list_of_nodes,tnow,thetacrit)

                    else:
                        node=self.nodes[node.sibling]
                        self.split_node(node,list_of_nodes,tnow,thetacrit)
                        node=self.nodes[node.parent]
                        if node.predecessor==-1:
                            notdone=False
                        else:
                            if node.sibling!=node.predecessor:
                                node=self.nodes[node.predecessor]
                                self.split_node(node,list_of_nodes,tnow,thetacrit)
                
        return list_of_nodes
    
    def split_node(self,node,list_of_nodes,tnow,thetacrit):
    
        '''
	    split_node
	   
	    Function which decides whether or not to split a given node into its children,
	    based on the critical angle and the current time
	    '''
    
        nodestocheck=[]
        nodestocheck.append(node)
        while len(nodestocheck)>0:
            node=nodestocheck.pop(0)
            mid=(node.start+node.end)/2.
            width=(node.end-node.start)
            age=tnow-mid
            theta=width/age

            if theta<thetacrit:
                list_of_nodes.append(node.nodeid)
            else:
                if len(node.children)>0:
                    nodestocheck.append(self.nodes[node.children[0]])
                    nodestocheck.append(self.nodes[node.children[1]])
                else:
                    list_of_nodes.append(node.nodeid)

    def make_weight_functions(self):
        self.weight_functions['uniform']=self._uniform
        self.weight_functions['oneovert']=self._oneovert
        self.weight_functions['sigmoid']=self._sigmoid
        self.weight_functions['expon']=self._expon
        self.weight_functions['ramp']=self._ramp
                    
    def _uniform(self,node_list,tnow,sigma=1):
        weights=np.ones(len(node_list))
        return weights

    def _oneovert(self,node_list,tnow,sigma=1):
        weights=np.zeros(len(node_list))
        for i in range(len(node_list)):
            mid=(self.nodes[node_list[i]].start+self.nodes[node_list[i]].end)/2.
            lag=tnow-mid+1.5
            weights[i]=1./lag
        return weights
    
    def _sigmoid(self,node_list,tnow,sigma=1):
        weights=np.zeros(len(node_list))
        offset=5.
        sigma/=offset
        for i in range(len(node_list)):
            mid=self.nodes[node_list[i]].start
            lag=(mid-tnow+offset*sigma)/sigma
            weights[i]=1./(1+np.exp(-lag))

        return weights
    
    def _expon(self,node_list,tnow,sigma=1):
        weights=np.zeros(len(node_list))
        for i in range(len(node_list)):
            mid=(self.nodes[node_list[i]].start+self.nodes[node_list[i]].end)/2.
            lag=tnow-mid
            lag1=tnow-self.nodes[node_list[i]].start
            
            lag2=tnow-self.nodes[node_list[i]].end
            
            w=np.exp(-lag/sigma)
            w1=np.exp(-lag1/sigma)
            w2=np.exp(-lag2/sigma)
            weights[i]=(8*w1+6*w-w2)/13.

        return weights
    
    def _ramp(self,node_list,tnow,sigma=1):
        weights=np.zeros(len(node_list))
        for i in range(len(node_list)):
            mid=(self.nodes[node_list[i]].start+self.nodes[node_list[i]].end)/2.
            lag=tnow-mid+0.5
            weights[i]=1.-(lag)/(sigma)

        weights[weights<0.0]=0.0
        return weights
    
    def tree_lag(self,df,thetacrit,weight_functions,sigma):
    
        '''
        tree_lag
        
        This function computes features temporally lagged using the tree for every pgid.
        
        The interaction lists and weights, and the features loaded into each node are 
        combined in weighted sums.
        
        The lagged features are then packed into a dataframe.
        
        '''
    
        ntimes=len(self.times)
        npgids=len(self.pgids)
        nfeatures=len(self.features)
        nfunctions=len(weight_functions)
    
        tensor3d=np.empty((ntimes,npgids,nfeatures*nfunctions))
    
        for time in self.times:
            itime=self.time_to_index[time]
            list_of_nodes=self.walk(time,thetacrit)
            for ifunction,weight_function in enumerate(weight_functions):
                weights=self.weight_functions[weight_function](list_of_nodes,time,sigma)
                for ifeature,feature in enumerate(self.features):
                    weighted_feature=np.zeros_like(self.nodes[0].features[feature])
                    for nodeid,weight in zip(list_of_nodes,weights):
                        weighted_feature+=weight*self.nodes[nodeid].features[feature]
        
                    tensor3d[itime,:,ifeature*nfunctions+ifunction]=weighted_feature
                    
        flat=np.empty((ntimes*npgids,nfeatures*nfunctions))
    
        for ichunk in range(ntimes):
            flat[ichunk*npgids:(ichunk+1)*npgids,:]=tensor3d[ichunk,:,:]
       
        df_column_names=['tree_tlag_'+function+'_'+feature for feature in self.features for function in weight_functions]
        
        index_names=df.index.names
        
        df_index=pd.MultiIndex.from_product([self.times, self.pgids],names=index_names)    
    
        df_treelags=pd.DataFrame(flat, index=df_index, columns=df_column_names)
    
        return df_treelags    