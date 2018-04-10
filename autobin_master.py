from autobin4woe import *
from autobin_quantile import *
from autobin_freq import *
  
def autobin_master(df,feature,label,Y_org,initial_bin_num,min_class_pct,stop_limit):
  ## select best discrimination binning method from 1: woe bin/2: quantile bin/3: frequency bin
  ## the initial bin number of woe bin is specified by parameter given by user,the other two's are set to 10 bins
  bindata_1,iv_1,binnum_1=autobin4woe(df,feature,label,initial_bin_num,min_class_pct,stop_limit)
  bindata_2,iv_2,binnum_2=autobin_quantile(df,feature,label,10,min_class_pct,stop_limit)
  bindata_3,iv_3,binnum_3=autobin_freq(df,feature,label,10,min_class_pct,stop_limit)
  bdt_1=pd.DataFrame(zip(bindata_1,Y_org),columns=['attribute','target'])
  bdt_2=pd.DataFrame(zip(bindata_2,Y_org),columns=['attribute','target'])
  bdt_3=pd.DataFrame(zip(bindata_3,Y_org),columns=['attribute','target'])
  bdt_1g=bdt_1.groupby(bdt_1['attribute'])['target'].mean()
  bdt_2g=bdt_2.groupby(bdt_2['attribute'])['target'].mean()
  bdt_3g=bdt_3.groupby(bdt_3['attribute'])['target'].mean()
  mm=[max(bdt_1g)/min(bdt_1g),max(bdt_2g)/min(bdt_2g),max(bdt_3g)/min(bdt_3g)]
  if np.argmax(mm)==0:
    print '%s : autobin4woe selected!' %feature ,
    print 'IV_value: %.4f, Bin_number: %i, Discrimination: %.4f' %(iv_1,binnum_1,mm[np.argmax(mm)])
    return bindata_1,iv_1,binnum_1,mm[np.argmax(mm)]
  elif np.argmax(mm)==1: 
    print '%s : autobin_quantile selected!' %feature ,
    print 'IV_value: %.4f, Bin_number: %i, Discrimination: %.4f' %(iv_2,binnum_2,mm[np.argmax(mm)])
    return bindata_2,iv_2,binnum_2,mm[np.argmax(mm)]
  else:
    print '%s : autobin_freq selected!' %feature ,
    print 'IV_value: %.4f, Bin_number: %i, Discrimination: %.4f' %(iv_3,binnum_3,mm[np.argmax(mm)])
    return bindata_3,iv_3,binnum_3,mm[np.argmax(mm)]

