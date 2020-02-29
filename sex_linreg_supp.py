#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:24:23 2018

Compares sumstats from different linreg models which have varying dependence on sex
Uses results from sex_linreg.py and sex_linreg_PCs.py

@author: nbaya
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats


"""
For nested linreg models:
    
    1) y ~ intercept + 20 PCs
    2) y ~ intercept + 20 PCs + sex 
    3) y ~ intercept + 20 PCs + age + age^2
    4) y ~ intercept + 20 PCs + sex + age + age^2
    5) y ~ intercept + 20 PCs + sex + age + sex*age
    6) y ~ intercept + 20 PCs + sex + age + sex*age + sex*age^2

"""

sex_linreg_wd = '/Users/nbaya/Documents/lab/ukbb-sexdiff/sex_linreg'

phsource_list = ['phesant','icd10','finngen']


reg_list = list(range(1,7))


"""
Copy all sumstats from gcloud
"""
for phsource in phsource_list:
    for reg in reg_list:
        reg = str(reg)
        reg_fname=f'ukb31063.{phsource}_phenotypes.both_sexes.reg{reg}.tsv.bgz'
        if not os.path.isfile(f'{sex_linreg_wd}/{reg_fname}'):
            os.system(f'gsutil cp gs://nbaya/sex_linreg/{reg_fname} {sex_linreg_wd}')

"""
Create pandas dataframes
"""
phsource_list = ['phesant']#, 'finngen']

def get_reg(reg, phsource):
    r'''
    Get regression results for regression model number `reg` and phsource `phsource`
    '''
    reg_fname=f'ukb31063.{phsource}_phenotypes.both_sexes.reg{reg}.tsv.bgz'
    reg_df = pd.read_csv(f'{sex_linreg_wd}/{reg_fname}', compression='gzip',sep='\t')
    reg_df.insert(0,'source',phsource) #put phsource in first column
    reg_df.insert(1,'reg',reg) #put regression index in second column
    return reg_df

reg_df_list = [[get_reg(reg=reg, phsource=phsource) for reg in reg_list] for phsource in phsource_list]

reg_df = pd.concat(*reg_df_list, ignore_index=True)

## get UKB round 2 h2 results
def get_h2(phsource):
    h2_results_path=f'/Users/nbaya/Documents/lab/ukbb-sexdiff/rg_sex/ukbb31063.both_sexes.h2part_results.{"v2." if phsource=="phesant" else ""}{phsource}.tsv.gz'
    h2 = pd.read_csv(h2_results_path, sep='\t',compression='gzip')
    h2 = h2.rename(index=str,columns={'phenotype':'phen'}).iloc[:,0:20]
    return h2


h2_list = [get_h2(phsource=phsource) for phsource in phsource_list]
h2 = pd.concat(h2_list, ignore_index=True) # dataframe with h2

## remove "_irnt" suffix from h2 phens with irnt suffix
#h2[h2.phen.str.contains('irnt')].phen #lists all phens with irnt suffix
h2['phen'] = h2.applymap(lambda x: str(x).replace('_irnt','')).phen 

## combine sumstats (reg1, reg2, reg3) with heritability results (h2)
keys = list(['source','phen'])
h2_idx = h2.set_index(keys).index

df = reg_df.merge(h2, on=keys)

df_tmp = df[df.reg==6].copy()
fields = [f.replace('beta_','') for f in df_tmp.columns.values if 'beta_' in f]
pcs = [f'PC{pc}' for pc in range(1,21)]
non_pcs = sorted([f for f in fields if 'PC' not in f])

def plot_qq(df_tmp, fields, gradient=True, logscale=False):
    r'''
    Makes QQ plot for fields in `fields`. Colors with gradient if `gradient`=True,
    otherwise uses default color cycle. Plots in log-log scale if `logscale`=True.
    '''
    df_tmp=df_tmp.copy()
    fig, ax = plt.subplots(figsize=(9*1.2,6*1.2))
    exp = -np.log10(np.linspace(start=1,stop=1/df_tmp.shape[0],num=df_tmp.shape[0])) # to account for weird kink at lower end in log-log scale: -np.log10(np.linspace(start=1-1/df_tmp.shape[0],stop=1/df_tmp.shape[0],num=df_tmp.shape[0]))
    n = len(df_tmp)
    k = np.arange(1,n+1)
    a = k
    b = n+1-k
    intervals=stats.beta.interval(0.95, a, b) # get 95% CI with parameters a and b
    ax.plot(exp,exp,'k--')
    plt.fill_between(x=exp[::-1], y1=-np.log10(intervals[0]), y2=-np.log10(intervals[1]), # need to reverse order of array exp to match intervals
                    color='k', alpha=0.1)
    
    for i, field in enumerate(fields):
        df_tmp[f'z_{field}'] = df_tmp[f'beta_{field}']/df_tmp[f'se_{field}']
        df_tmp[f'nlog10p_{field}'] = -np.log10(2) - stats.norm.logcdf(-abs(df_tmp[f'z_{field}']))/np.log(10)
        
        rank = np.sort(df_tmp[f'nlog10p_{field}'])
        if gradient:
            ax.plot(exp, rank, color=plt.cm.jet(i/len(fields)))
        else:
            ax.plot(exp, rank)
    if logscale:
        plt.yscale('log')
        plt.xscale('log')
#    plt.ylim([min(-np.log10(intervals[0])),max(-np.log10(intervals[1]))])
#    plt.xlim([-0.002,0.002])
    plt.xlabel(r'expected -$log_{10}(p)$')
    plt.ylabel(r'observed -$log_{10}(p)$')
    plt.legend(['expected']+fields, loc=2)
    
    return df_tmp


df_tmp_pcs = plot_qq(df_tmp=df_tmp, fields=pcs, logscale=True)
df_tmp_nonpcs = plot_qq(df_tmp=df_tmp, fields=[f for f in non_pcs if f!='intercept'],
                        gradient=False, logscale=True)

#for field in fields:
#    print(f'\n--------- {field} ---------')
#    print(df_tmp_nonpcs.sort_values(by=f'nlog10p_{field}', ascending=False)[
#            ['phen','description',f'z_{field}',f'nlog10p_{field}']].head(10))





"""
Analyze results
"""
phsource='phesant'
#get phenotypes with top 10 r2_mul
df[(df['reg']==3)&(df['source']==phsource)].sort_values(by='r2_mul',ascending=False)[
        ['phen','description','r2_mul','r2_adj','h2_observed']].head(10)

#get phenotypes with top 10 r2_mul, with h2 threshold
h2_thresh = 0.1
df[(df['reg']==2)&(df['source']==phsource)&(df['h2_observed']>h2_thresh)].sort_values(by='r2_mul',ascending=False)[
        ['phen','description','r2_mul','r2_adj','h2_observed']].head(10)

#phenotypes with highest difference in reg1 r2_mul vs reg2 r2_mul
all_reg[all_reg['source']==phsource].sort_values(by=['r2_mul_reg1_minus_reg2'],ascending=False)[['phen','r2_mul_reg1','r2_mul_reg2','r2_mul_reg1_minus_reg2']]

#phenotypes with highest difference in reg2 r2_mul vs reg3 r2_mul
all_reg[all_reg['source']==phsource].sort_values(by=['r2_mul_reg2_minus_reg3'],ascending=False)[['phen','r2_mul_reg2','r2_mul','r2_mul_reg2_minus_reg3']]

#phenotypes with highest difference in reg1 r2_mul vs reg3 r2_mul
all_reg[all_reg['source']==phsource].sort_values(by=['r2_mul_reg1_minus_reg3'],ascending=False)[['phen','r2_mul_reg1','r2_mul','r2_mul_reg1_minus_reg3']]


phsource='phesant'
df[(df['source']==phsource) & (df['reg']=='reg3')].h2_observed

sexdiff=pd.read_csv('/Users/nbaya/Documents/Lab/ukbb-sexdiff/imputed-v3-results/ukbb31063.rg_sex.v1.csv')
sexdiff=sexdiff.rename(index=str,columns={'phenotype':'phen'})
reg_sexdiff=all_reg.merge(sexdiff, on='phen')
reg_sexdiff['abs_h2_diff'] = abs(reg_sexdiff.ph1_h2_obs-reg_sexdiff.ph2_h2_obs)
reg_sexdiff['fm_h2_diff'] = reg_sexdiff.ph1_h2_obs-reg_sexdiff.ph2_h2_obs #female_h2_obs - male_h2_obs






"""
Plot results
"""
phsource = 'finngen'
#plot r2_adj vs h2_observed
for i in list(map(str,range(1,4))):
    plt.plot(df[(df['reg']=='reg'+i) & (df['source']==phsource)].r2_adj,df[(df['reg']=='reg'+i) & (df['source']==phsource)].h2_observed,'.')
plt.legend(['reg1','reg2','reg3'])
plt.ylabel('h2_observed')
plt.xlabel('r2_adj')
plt.title('r2_adj vs. h2_observed\n('+phsource+' phenotypes)')
fig = plt.gcf()
fig.set_size_inches(6*1.5, 4*1.5)
fig.savefig('/Users/nbaya/Desktop/r2_adj_h2_observed_'+phsource+'.png',dpi=300)

#plot r2_adj vs intercept
for i in list(map(str,range(1,4))):
    plt.plot(df[(df['reg']=='reg'+i) & (df['source']==phsource)].r2_adj,df[(df['reg']=='reg'+i) & (df['source']==phsource)].intercept,'.')
plt.legend(['reg1','reg2','reg3'])
plt.ylabel('intercept')
plt.xlabel('r2_adj')
plt.title('r2_adj vs. intercept\n('+phsource+' phenotypes)')
fig = plt.gcf()
fig.set_size_inches(6*1.5, 4*1.5)
fig.savefig('/Users/nbaya/Desktop/r2_adj_intercept_'+phsource+'.png',dpi=300)

#plot kde plots r2_mul or r2_adj
r2_suf = '_mul'
for i in list(map(str,range(1,4))):
    sns.kdeplot(df[(df['reg']=='reg'+i) & (df['source']==phsource)]['r2'+r2_suf])
plt.text(0.45,8,'mean reg1.r2'+r2_suf+': %.4f' % np.mean(comb1[comb1['source']==phsource]['r2'+r2_suf]))
plt.text(0.45,7,'mean reg2.r2'+r2_suf+': %.4f' % np.mean(comb2[comb2['source']==phsource]['r2'+r2_suf]))
plt.text(0.45,6,'mean reg3.r2'+r2_suf+': %.4f' % np.mean(comb3[comb3['source']==phsource]['r2'+r2_suf]))
plt.text(0.45,5,'std reg1.r2'+r2_suf+': %.4f' % np.std(comb1[comb1['source']==phsource]['r2'+r2_suf]))
plt.text(0.45,4,'std reg2.r2'+r2_suf+': %.4f' % np.std(comb2[comb2['source']==phsource]['r2'+r2_suf]))
plt.text(0.45,3,'std reg3.r2'+r2_suf+': %.4f' % np.std(comb3[comb3['source']==phsource]['r2'+r2_suf]))
plt.legend(['reg1','reg2','reg3'])
plt.title('kde plots of r2'+r2_suf+'\n('+phsource+' phenotypes)')
fig = plt.gcf()
fig.set_size_inches(6*1.5,4*1.5)
fig.savefig('/Users/nbaya/Desktop/r2'+r2_suf+'_kde_'+phsource+'.png',dpi=300)


#plot r2_mul for reg1 vs reg2, reg2 vs reg 3, and reg1 vs reg3
plt.plot([0,1],[0,1],'k',linewidth=0.5)
plt.plot(all_reg[all_reg['source']==phsource][['r2_adj_reg1']],all_reg[all_reg['source']==phsource][['r2_adj_reg2']],'+',markersize=5)
plt.plot(all_reg[all_reg['source']==phsource][['r2_adj_reg2']],all_reg[all_reg['source']==phsource][['r2_adj']],'+',markersize=5)
plt.plot(all_reg[all_reg['source']==phsource][['r2_adj_reg1']],all_reg[all_reg['source']==phsource][['r2_adj']],'+',markersize=5)
axes_lim = np.min([np.max(all_reg[all_reg['source']==phsource][['r2_adj_reg1']])[0]*1.5,1])
plt.ylim([0, axes_lim])
plt.xlim([0, axes_lim])
plt.legend(['y=x','r2_adj reg1 vs r2_adj reg2','r2_adj reg2 vs r2_adj reg3','r2_adj reg1 vs r2_adj reg3'])
plt.xlabel('r2_adj')
plt.ylabel('r2_adj')
if (all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj_reg2']>all_reg[all_reg['source']==phsource]['r2_adj_reg1']].shape[0] == 0 and
    all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj']>all_reg[all_reg['source']==phsource]['r2_adj_reg2']].shape[0] == 0 and
    all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj']>all_reg[all_reg['source']==phsource]['r2_adj_reg1']].shape[0] == 0):
    plt.text(0.02*axes_lim,0.70*axes_lim,' Note: None of the points lie above  \n           the y=x line.',
             fontsize=9, bbox=dict(facecolor='red',alpha=0.2))
else:
    plt.plot(all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj_reg2']>all_reg[all_reg['source']==phsource]['r2_adj_reg1']]['r2_adj_reg1'],all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj_reg2']>all_reg[all_reg['source']==phsource]['r2_adj_reg1']][['r2_adj_reg2']],'r+',markersize=5)
    plt.plot(all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj']>all_reg[all_reg['source']==phsource]['r2_adj_reg2']]['r2_adj_reg2'],all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj']>all_reg[all_reg['source']==phsource]['r2_adj_reg2']][['r2_adj']],'r+',markersize=5)
    plt.plot(all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj']>all_reg[all_reg['source']==phsource]['r2_adj_reg1']]['r2_adj_reg1'],all_reg[all_reg['source']==phsource][all_reg[all_reg['source']==phsource]['r2_adj']>all_reg[all_reg['source']==phsource]['r2_adj_reg1']][['r2_adj']],'r+',markersize=5)
    plt.legend(['y=x','r2_adj reg1 vs r2_adj reg2','r2_adj reg2 vs r2_adj reg3','r2_adj reg1 vs r2_adj reg3','points above y=x'])
plt.title('r2_adj comparison\n('+phsource+' phenotypes)')
fig = plt.gcf()
fig.set_size_inches(6*1.5,4*1.5)
fig.savefig('/Users/nbaya/Desktop/r2_adj_comparison_'+phsource+'.png',dpi=300)



#plot histograms
bins=list(np.linspace(0,0.6,13))
plt.subplot(3,1,1)
plt.hist(comb1[comb1['source']==phsource].r2_adj,bins=bins)
plt.xlim([0,.6])
plt.ylim([0,3000])
plt.text(0.35,2500+275,'mean reg1.r2_adj: %.4f' % np.mean(comb1[comb1['source']==phsource].r2_adj))
plt.text(0.35,2250+275,'std reg1.r2_adj: %.4f' % np.std(comb1[comb1['source']==phsource].r2_adj))
plt.title('Histogram of r2 values for regression 1\n('+phsource+' phenotypes)')

plt.subplot(3,1,2)
plt.hist(comb2[comb2['source']==phsource].r2_adj,bins=bins)
plt.xlim([0,.6])
plt.ylim([0,3000])
plt.text(0.35,2500+275,'mean reg2.r2_adj: %.4f' % np.mean(comb2[comb2['source']==phsource].r2_adj))
plt.text(0.35,2250+275,'std reg2.r2_adj: %.4f' % np.std(comb2[comb2['source']==phsource].r2_adj))
plt.title('Histogram of r2 values for regression 2\n('+phsource+' phenotypes)')

plt.subplot(3,1,3)
plt.hist(comb3[comb3['source']==phsource].r2_adj,bins=bins)
plt.xlim([0,.6])
plt.ylim([0,3000])
plt.text(0.35,2500+275,'mean reg3.r2_adj: %.4f' % np.mean(comb3[comb3['source']==phsource].r2_adj))
plt.text(0.35,2250+275,'std reg3.r2_adj: %.4f' % np.std(comb3[comb3['source']==phsource].r2_adj))
plt.title('Histogram of r2 values for regression 3\n('+phsource+' phenotypes)')
plt.tight_layout(pad=0.01)
fig = plt.gcf()
fig.set_size_inches(4*1.5,6*2)

#plot differences in r2_mul across regression models
plt.plot(list(range(all_reg.shape[0])),all_reg.sort_values(by=['r2_mul_reg1_minus_reg2'],ascending=False)['r2_mul_reg1_minus_reg2'],'.')
plt.plot(list(range(all_reg.shape[0])),all_reg.sort_values(by=['r2_mul_reg2_minus_reg3'],ascending=False)['r2_mul_reg2_minus_reg3'],'.')
plt.plot(list(range(all_reg.shape[0])),all_reg.sort_values(by=['r2_mul_reg1_minus_reg3'],ascending=False)['r2_mul_reg1_minus_reg3'],'.')

#sns.kdeplot(all_reg['r2_mul_reg2_minus_reg3'])
#sns.kdeplot(all_reg['r2_mul_reg1_minus_reg3'])
plt.xlim([0,0.2])
np.mean(all_reg['r2_mul_reg1_minus_reg2'])
np.mean(all_reg['r2_mul_reg2_minus_reg3'])
np.mean(all_reg['r2_mul_reg1_minus_reg3'])
plt.stackplot(list(range(all_reg.shape[0])),list(map(lambda x: -1/np.log(x+10e-20),[all_reg.sort_values(by=['r2_mul_reg1_minus_reg2'],ascending=False)['r2_mul_reg1_minus_reg2'],
              all_reg.sort_values(by=['r2_mul_reg2_minus_reg3'],ascending=False)['r2_mul_reg2_minus_reg3']])))
fig = plt.gcf()
fig.set_size_inches(6*1.5,4*1.5)

plt.plot(all_reg[all_reg['r2_mul']<=all_reg['r2_mul_reg1']][['r2_mul_reg1']],all_reg[all_reg['r2_mul']<=all_reg['r2_mul_reg1']][['r2_mul']],'.')
plt.plot(all_reg[all_reg['r2_mul']>all_reg['r2_mul_reg1']][['r2_mul_reg1']],all_reg[all_reg['r2_mul']>all_reg['r2_mul_reg1']][['r2_mul']],'.')
plt.xlabel('r2_mul reg1')
plt.ylabel('r2_mul reg3')

plt.plot([0,0.5],[0,0.5],'k')
#plt.plot(all_reg[['r2_adj_reg1']],all_reg[['r2_adj']],'.')
plt.plot(all_reg[all_reg['r2_adj']<=all_reg['r2_adj_reg1']][['r2_adj_reg1']],all_reg[all_reg['r2_adj']<=all_reg['r2_adj_reg1']][['r2_adj']],'.')
plt.plot(all_reg[all_reg['r2_adj']>all_reg['r2_adj_reg1']][['r2_adj_reg1']],all_reg[all_reg['r2_adj']>all_reg['r2_adj_reg1']][['r2_adj']],'.')
plt.xlabel('r2_adj reg1')
plt.ylabel('r2_adj reg3')

for phen in all_reg[all_reg['source']==phsource].phen:
    if comb3[comb3['source']==phsource][comb3['phen']==phen].shape[0] != 0:
        plt.plot([all_reg[all_reg['source']==phsource][all_reg['phen']==phen].r2_adj_reg1.iloc[0],
                  all_reg[all_reg['source']==phsource][all_reg['phen']==phen].r2_adj_reg2.iloc[0], 
                  all_reg[all_reg['source']==phsource][all_reg['phen']==phen].r2_adj.iloc[0]],
        [comb3[comb3['source']==phsource][comb3['phen']==phen].h2_observed.iloc[0],
         comb3[comb3['source']==phsource][comb3['phen']==phen].h2_observed.iloc[0],
         comb3[comb3['source']==phsource][comb3['phen']==phen].h2_observed.iloc[0]],'k.-')
        

#plot differences in r2_mul against male/female rg
plt.subplot(1,3,1)
plt.plot(reg_sexdiff.rg, reg_sexdiff.r2_mul_reg1_minus_reg2,'.')
#plt.plot(reg_sexdiff[reg_sexdiff.p1.str.contains('1588')].rg,
#                     reg_sexdiff[reg_sexdiff.p1.str.contains('1588')].r2_mul_reg1_minus_reg2,'r.')

plt.xlabel('male/female rg')
plt.ylabel('reg1 R2 - reg2 R2')
plt.title('male/female rg vs. reg1 R2 - reg2 R2')

plt.subplot(1,3,2)
plt.plot(reg_sexdiff.rg, reg_sexdiff.r2_mul_reg2_minus_reg3,'.')
plt.plot(reg_sexdiff[reg_sexdiff.p1.str.contains('30010')].rg,
                     reg_sexdiff[reg_sexdiff.p1.str.contains('30010')].r2_mul_reg2_minus_reg3,'r.')
plt.plot(reg_sexdiff[reg_sexdiff.p1.str.contains('30180')].rg,
                     reg_sexdiff[reg_sexdiff.p1.str.contains('30180')].r2_mul_reg2_minus_reg3,'y.')
plt.xlabel('male/female rg')
plt.ylabel('reg3 R2 - reg3 R2')
plt.title('male/female rg vs. reg2 R2 - reg3 R2')

plt.subplot(1,3,3)
plt.plot(reg_sexdiff.rg, reg_sexdiff.r2_mul_reg1_minus_reg3,'.')
plt.plot(reg_sexdiff[reg_sexdiff.p1.str.contains('30010')].rg,
                     reg_sexdiff[reg_sexdiff.p1.str.contains('30010')].r2_mul_reg1_minus_reg3,'r.')
plt.plot(reg_sexdiff[reg_sexdiff.p1.str.contains('30180')].rg,
                     reg_sexdiff[reg_sexdiff.p1.str.contains('30180')].r2_mul_reg1_minus_reg3,'y.')
plt.xlabel('male/female rg')
plt.ylabel('reg1 R2 - reg3 R2')
plt.title('male/female rg vs. reg1 R2 - reg3 R2')

fig = plt.gcf()
fig.set_size_inches(18,6)
fig.savefig('/Users/nbaya/Desktop/sexdiff_rg_R2diff.png',dpi=300)


#plot differences in r2_mul against absolute m/f difference in h2_observed
plt.subplot(1,3,1)
plt.plot(reg_sexdiff.abs_h2_diff, reg_sexdiff.r2_mul_reg1_minus_reg2,'.')
plt.xlabel('absolute h2_obs m/f difference')
plt.ylabel('reg1 R2 - reg2 R2')
plt.title('absolute female/male h2_obs diff vs. reg1 R2 - reg2 R2')

plt.subplot(1,3,2)
plt.plot(reg_sexdiff.abs_h2_diff, reg_sexdiff.r2_mul_reg2_minus_reg3,'.')
plt.xlabel('absolute h2_obs m/f difference')
plt.ylabel('reg3 R2 - reg3 R2')
plt.title('absolute female/male h2_obs diff vs. reg2 R2 - reg3 R2')

plt.subplot(1,3,3)
plt.plot(reg_sexdiff.abs_h2_diff, reg_sexdiff.r2_mul_reg1_minus_reg3,'.')
plt.xlabel('absolute h2_obs m/f difference')
plt.ylabel('reg1 R2 - reg3 R2')
plt.title('absolute female/male h2_obs diff vs. reg1 R2 - reg3 R2')
fig = plt.gcf()
fig.set_size_inches(18,6)
fig.savefig('/Users/nbaya/Desktop/abs_sexdiff_h2_R2diff.png',dpi=300)

#plot differences in r2_mul against f-m difference in h2_observed
plt.subplot(1,3,1)
plt.plot(reg_sexdiff[reg_sexdiff.fm_h2_diff>0].fm_h2_diff, reg_sexdiff[reg_sexdiff.fm_h2_diff>0].r2_mul_reg1_minus_reg2,'r.')
plt.plot(reg_sexdiff[reg_sexdiff.fm_h2_diff<0].fm_h2_diff, reg_sexdiff[reg_sexdiff.fm_h2_diff<0].r2_mul_reg1_minus_reg2,'b.')
plt.ylim([-0.00015,0.012])
plt.legend(['female h2_obs > male h2_obs','female h2_obs < male h2_obs'])
plt.xlabel('female h2_obs - male h2_obs')
plt.ylabel('reg1 R2 - reg2 R2')
plt.title('female/male h2_obs diff vs. reg1 R2 - reg2 R2')

plt.subplot(1,3,2)
plt.plot(reg_sexdiff[reg_sexdiff.fm_h2_diff>0].fm_h2_diff, reg_sexdiff[reg_sexdiff.fm_h2_diff>0].r2_mul_reg2_minus_reg3,'r.')
plt.plot(reg_sexdiff[reg_sexdiff.fm_h2_diff<0].fm_h2_diff, reg_sexdiff[reg_sexdiff.fm_h2_diff<0].r2_mul_reg2_minus_reg3,'b.')
plt.ylim([-0.01,0.72])
plt.legend(['female h2_obs > male h2_obs','female h2_obs < male h2_obs'])
plt.xlabel('female h2_obs - male h2_obs')
plt.ylabel('reg2 R2 - reg3 R2')
plt.title('female/male h2_obs diff vs. reg2 R2 - reg3 R2')

plt.subplot(1,3,3)
plt.plot(reg_sexdiff[reg_sexdiff.fm_h2_diff>0].fm_h2_diff, reg_sexdiff[reg_sexdiff.fm_h2_diff>0].r2_mul_reg1_minus_reg3,'r.')
plt.plot(reg_sexdiff[reg_sexdiff.fm_h2_diff<0].fm_h2_diff, reg_sexdiff[reg_sexdiff.fm_h2_diff<0].r2_mul_reg1_minus_reg3,'b.')
plt.ylim([-0.01,0.72])
plt.legend(['female h2_obs > male h2_obs','female h2_obs < male h2_obs'])
plt.xlabel('female h2_obs - male h2_obs')
plt.ylabel('reg1 R2 - reg3 R2')
plt.title('female/male h2_obs diff vs. reg2 R1 - reg3 R2')

fig = plt.gcf()
fig.set_size_inches(18,6)
fig.savefig('/Users/nbaya/Desktop/sexdiff_h2_R2diff.png',dpi=300)
###############################################################################
"""
For linreg model of sex as the response variable and PCs + age + age^2 as the 
covariates.
"""
local_wd = '/Users/nbaya/Documents/lab/ukbb-sexdiff/sex_linreg/'
filename = 'ukb31063.both_sexes.sex_linreg_w_PCs.tsv.bgz'
if not os.path.isfile(local_wd+filename):
    os.system('gsutil cp gs://nbaya/sex_linreg/'+filename+' '+local_wd)

sex_reg = pd.read_csv(local_wd+filename, sep='\t',compression='gzip')

sex_reg

sex_reg.filter(regex=('pval_PC'))
