# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from compiler.ast import flatten


def autobin4woe(df,feature,label,initial_bin_num,min_class_pct,stop_limit):
    df_bk=df[[feature,label]] # used for final binning
    df=df_bk.copy()           # used for iterative merging of bins
    df.columns=['predictor','target']

# Binning in case a numerical variable was selected
#    if(len(np.unique(df.target))==2 and df.predictor.dtypes=float64 ):
        # Derive cutpoints for bins
    cutpoints=[min(df.predictor)]
    quantile_value=df.predictor.quantile(map(lambda x:1.0*x/initial_bin_num,\
                                             range(1,initial_bin_num)),\
                                                interpolation='nearest')
    cutpoints.append(list(quantile_value))
    cutpoints.append(max(df.predictor))
    cutpoints=flatten(cutpoints)
    cutpoints=list(np.unique(cutpoints)) # remove multiple cutpoints with same value
    #print "Initial binning number is: %i" %(len(cutpoints)-1)   
    ## Calculate initial crosstab from binned variable and target variable
    df.predictor_bin=pd.cut(df.predictor,cutpoints, right=True,\
                        labels=range(len(cutpoints)-1), include_lowest=True)
    # Compute crosstab from binned variable and target variable
    cttab=pd.crosstab(df.predictor_bin,df.target)
    cttab.columns=['good','bad']
    # Compute columns percents for target classes from crosstab frequencies
    cttab['good_ratio']=1.0*cttab.ix[:,'good']/sum(cttab.ix[:,'good'])
    cttab['bad_ratio']=1.0*cttab.ix[:,'bad']/sum(cttab.ix[:,'bad'])
    # Correct column percents in case of 0 frequencies 
    if (min(cttab.good)==0 or min(cttab.bad)==0):
        cttab.good_ratio=1.0*(cttab.good_ratio+0.0001)/sum(cttab.good_ratio+0.0001)
        cttab.bad_ratio=1.0*(cttab.bad_ratio+0.0001)/sum(cttab.bad_ratio+0.0001)

    # Check for bins if frequencies < percentage limit specified above
    # (in reverse order to remain correct reference to cutpoints)

    for i in xrange(len(cttab)-1,1,-1):
        if (cttab.ix[i,'good_ratio']<min_class_pct or \
            cttab.ix[i,'bad_ratio']<min_class_pct or \
            1.0*(cttab.ix[i,'good']+cttab.ix[i,'bad'])/sum(cttab.ix[:,'good']+cttab.ix[:,'bad'])<1.0/initial_bin_num):
            ## remove cutpoint
            del cutpoints[i]
            ## Compute binned variable from cutpoints and add it to the subset data frame
            df.predictor_bin=pd.cut(df.predictor,cutpoints, right=True,\
                                labels=range(len(cutpoints)-1), include_lowest=True)
            # Compute crosstab from binned variable and target variable
            cttab=pd.crosstab(df.predictor_bin,df.target)
            cttab.columns=['good','bad']
            # Compute columns percents for target classes from crosstab frequencies
            cttab['good_ratio']=1.0*cttab.ix[:,'good']/sum(cttab.ix[:,'good'])
            cttab['bad_ratio']=1.0*cttab.ix[:,'bad']/sum(cttab.ix[:,'bad'])
            # Correct column percents in case of 0 frequencies 
            if (min(cttab.good)==0 or min(cttab.bad)==0):
                cttab.good_ratio=1.0*(cttab.good_ratio+0.0001)/sum(cttab.good_ratio+0.0001)
                cttab.bad_ratio=1.0*(cttab.bad_ratio+0.0001)/sum(cttab.bad_ratio+0.0001)
        # Stop in case 3 cutpoints  are reached
        if (len(cutpoints)==3):  break 
    cttab['woe100']=100.0*np.log(cttab['good_ratio']/cttab['bad_ratio'])           
    tmp=list(abs(np.diff(cttab['woe100'])))
    tmp.append(0)
    cttab['woe100_diff']=tmp
    del tmp
    cttab['iv']=(cttab['good_ratio']-cttab['bad_ratio'])*(cttab['woe100'])/100.0
#    print len(cttab)
    ## Merge bins with similar WOE values and calculate corresponding WOE table and IV step by step
    ## until 2 bins are left (i.e. 3 cutpoints: min, middle cutpoint, max)
    stop_limit_exceeded=False
    while(len(cutpoints)>2 and stop_limit_exceeded==False):
        ## Compute binned variable from cutpoints and add it to the subset data frame
        df.predictor_bin=pd.cut(df.predictor,cutpoints, right=True,\
                            labels=range(len(cutpoints)-1), include_lowest=True)
        # Compute crosstab from binned variable and target variable
        cttab=pd.crosstab(df.predictor_bin,df.target)
        cttab.columns=['good','bad']
        # Compute columns percents for target classes from crosstab frequencies
        cttab['good_ratio']=1.0*cttab.ix[:,'good']/sum(cttab.ix[:,'good'])
        cttab['bad_ratio']=1.0*cttab.ix[:,'bad']/sum(cttab.ix[:,'bad'])
        # Correct column percents in case of 0 frequencies 
        if (min(cttab.good)==0 or min(cttab.bad)==0):
            cttab.good_ratio=1.0*(cttab.good_ratio+0.0001)/sum(cttab.good_ratio+0.0001)
            cttab.bad_ratio=1.0*(cttab.bad_ratio+0.0001)/sum(cttab.bad_ratio+0.0001)
        ## woe & iv calculation
        cttab['woe100']=100.0*np.log(cttab['good_ratio']/cttab['bad_ratio'])
        tmp=list(abs(np.diff(cttab['woe100'])))
        tmp.append(0)
        cttab['woe100_diff']=tmp
        del tmp
        cttab['iv']=(cttab['good_ratio']-cttab['bad_ratio'])*(cttab['woe100'])/100.0
        # Calculate total IV for current binning
        iv_total = sum(cttab['iv'])     
        # Collect total IVs for different binning solutions
        if locals().has_key('iv_collect'):
            iv_collect.append(iv_total)
        else:
            iv_collect=[iv_total]
        # In case IV decreases by more than percentage specified by stop.limit parameter above
        # restore former binning solution (cutpoints) and leave loop 
        if len(iv_collect)>1 and stop_limit_exceeded==False:
            iv_decrease=1.0*(iv_collect[len(iv_collect)-2]-iv_collect[len(iv_collect)-1])/iv_collect[len(iv_collect)-2]
#            print iv_decrease
            if iv_decrease>stop_limit:
                cutpoints_final=cutpoints_backup[:] #copy 
                cttab_final=cttab_backup.copy(deep=True)
                stop_limit_exceeded=True # indicates that stop limit is exceeded to prevent overriding the final solution
        # Save first cutpoint solution and corresponding WOE values as final solution 
        # (is used in case no WOE merging will be applied)
#        print stop_limit_exceeded
        if not locals().has_key('cutpoints_backup'):
            cutpoints_final=cutpoints[:]
            cttab_final=cttab.copy(deep=True)
        # Saves binning solution after last merging step in case the IV stop limit was not exceeded    
        if stop_limit_exceeded==False and len(cutpoints)==3:
            cutpoints_final=cutpoints[:]
            cttab_final=cttab.copy(deep=True)                
        ## Save backups of current cutpoints and corresponding WOE values 
        ## before merging to be able to retrieve solution in case IV decrease is too strong
        cutpoints_backup = cutpoints[:]
        cttab_backup = cttab.copy(deep=True)
        # Determine the index of the minimum WOE difference between adjacent bins and
        # merge bins with minimum WOE difference 
        min_woe_diff_idx=list(cttab['woe100_diff']).index(min(cttab['woe100_diff'][:-1]))
        del cutpoints[min_woe_diff_idx+1]
        
        
    ## Compute final IV
    iv_total_final=sum(cttab_final['iv'])
    
    ## Output final class
    predictor_autobin_final=pd.cut(df.predictor,cutpoints_final, right=True,\
                            labels=range(len(cutpoints_final)-1), include_lowest=True)
    predictor_autobin_final.columns=[feature]
    ## output woe dict
    woe_dict = {}
    for item in set(cttab_final.index):
#        print item
        woe_dict[item]=cttab_final.ix[item,'woe100']/100.0
    #print "Final binning number is  : %i" %(len(cutpoints_final)-1)  
    #print "Final IV value is: %f" %iv_total_final
    #print "Auto binning for WOE finished!\n"
    
    #return cttab_final,iv_total_final,cutpoints_final ## final woe check
    return predictor_autobin_final,iv_total_final,len(cutpoints_final)-1

    
    
def woe_transform( X, woe_dict):
    temp = np.copy(X).astype(float)
    for k in woe_dict.keys():
        woe = woe_dict[k]
        temp[np.where(temp == k)[0]] = woe * 1.0
    predictor_woe_final=pd.DataFrame(temp)
    return predictor_woe_final
