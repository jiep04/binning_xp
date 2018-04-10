# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from compiler.ast import flatten


def autobin4woe_n(df,feature,label,initial_bin_num,min_class_pct,stop_limit):
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


    ## Tree-based iterative partitioning of bins until IV based stop criteria is reached
    ## or 2 aggregated bins are left (i.e. 3 cutpoints: min, middle cutpoint, max).
    innercutpoints=cutpoints[1:len(cutpoints)-1]
    if len(cutpoints)>2:
      for i in range(len(innercutpoints)-1):
        for i in range(len(innercutpoints)):
          if locals().has_key('selected_cuts'):
            df.predictor_bin=pd.cut(df.predictor,list(np.sort(flatten([-np.inf,selected_cuts,innercutpoints[i],np.inf]))),right=True,\
                           labels=range(len(list(np.sort(flatten([-np.inf,selected_cuts,innercutpoints[i],np.inf]))))-1), include_lowest=True)
          else:
            df.predictor_bin=pd.cut(df.predictor,list(np.sort(flatten([-np.inf,innercutpoints[i],np.inf]))), right=True,\
                           labels=range(len(list(np.sort(flatten([-np.inf,innercutpoints[i],np.inf]))))-1), include_lowest=True)
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
            
        # Restore former solution in case stop criteria is reached and exit loop
        if locals().has_key('max_iv_collect_bk'):
          if max_iv_collect_bk*(1+stop_limit)>max(iv_collect):
            innercutpoints=innercutpoints_bk 
            break
        # Backups to be able to restore former solution in case stop criteria is reached
        max_iv_collect_bk = max(iv_collect)
        innercutpoints_bk = innercutpoints
        
        # Get index of cutpoint with highest IV and reset iv_collect
        index_optimal = iv_collect.index(max(iv_collect))
        iv_collect =[]
        
        # collect and sort selected cuts
        if locals().has_key('selected_cuts'):
          selected_cuts.append(innercutpoints[index_optimal])
        else:
          selected_cuts=[innercutpoints[index_optimal]]
        selected_cuts=list(np.unique(np.sort(selected_cuts)))
        
        # Remove selected cutpoint from cutpoint list
        del innercutpoints[index_optimal]

    try:
      ## Output final class
      cutpoints_final=list(np.sort(flatten([-np.inf,selected_cuts,np.inf])))
      predictor_autobin_final=pd.cut(df.predictor,cutpoints_final, right=True,\
                          labels=range(len(cutpoints_final)-1), include_lowest=True)
      predictor_autobin_final.columns=[feature]
      ## Compute final cttab 
      cttab_final=pd.crosstab(predictor_autobin_final,df.target)
      cttab_final.columns=['good','bad']
      cttab_final['good_ratio']=1.0*cttab_final.ix[:,'good']/sum(cttab_final.ix[:,'good'])
      cttab_final['bad_ratio']=1.0*cttab_final.ix[:,'bad']/sum(cttab_final.ix[:,'bad'])
      if (min(cttab_final.good)==0 or min(cttab_final.bad)==0):
          cttab_final.good_ratio=1.0*(cttab_final.good_ratio+0.0001)/sum(cttab_final.good_ratio+0.0001)
          cttab_final.bad_ratio=1.0*(cttab_final.bad_ratio+0.0001)/sum(cttab_final.bad_ratio+0.0001)
      ## final woe & iv calculation
      cttab_final['woe100']=100.0*np.log(cttab_final['good_ratio']/cttab_final['bad_ratio'])
      tmp=list(abs(np.diff(cttab_final['woe100'])))
      tmp.append(0)
      cttab_final['woe100_diff']=tmp
      del tmp
      cttab_final['iv']=(cttab_final['good_ratio']-cttab_final['bad_ratio'])*(cttab_final['woe100'])/100.0
       
      ## Compute final total IV
      iv_total_final=sum(cttab_final['iv'])
      
      ## output woe dict
      woe_dict = {}
      for item in set(cttab_final.index):
          woe_dict[item]=cttab_final.ix[item,'woe100']/100.0
      #print "Final binning number is  : %i" %(len(cutpoints_final)-1)  
      #print "Final IV value is: %f" %iv_total_final
      #print "Auto binning for WOE finished!\n"
      
      #return cttab_final,iv_total_final,cutpoints_final ## final woe check
      return predictor_autobin_final,iv_total_final,len(cutpoints_final)-1
    except UnboundLocalError,e:
      print e.message,', please check if %s is suit for numerical binning or try increassing initial_bin_num' %feature
      return df.predictor,np.nan,initial_bin_num

    
    
def woe_transform( X, woe_dict):
    temp = np.copy(X).astype(float)
    for k in woe_dict.keys():
        woe = woe_dict[k]
        temp[np.where(temp == k)[0]] = woe * 1.0
    predictor_woe_final=pd.DataFrame(temp)
    return predictor_woe_final
