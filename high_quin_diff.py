#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:36:05 2022

@author: emmabarash
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
import seaborn as sns
import glob
import random
import inflect
import scipy.stats
import re

if os.sep == '/':
    directory = '/Users/emmabarash/Lab/data'
else:
    directory = r'C:\Users\Emma_PC\Documents\data'

filelist = glob.glob(os.path.join(directory,'*','*.csv'))

finaldf = pd.DataFrame(columns = ['Time', 'Poke1', 'Poke2', 'Line1', 'Line2', 'Line3', 'Line4', 'Cue1',
       'Cue2', 'Cue3', 'Cue4', 'TasteID', 'AnID', 'Date', 'Taste_Delivery',
       'Delivery_Time', 'Latencies'])
filelist.sort()

for f in range(len(filelist)):
    df = pd.read_csv(filelist[f])
    group = df
    col = ['Line1', 'Line2']
    
    def parse_edges(group,col):
        delivery_idx = []
        group['TasteID'] = None
        group['AnID'] = filelist[f][-22:-18]
        group['Date'] = filelist[f][-17:-11]
        for j in col:
            col = j
            if col == 'Line1': 
                taste = 'suc'
            if col == 'Line2':
                taste = 'qhcl'
                
            cols = ['Time']+[col]
            data = group[cols]
            try: edges = data[data[col].diff().fillna(False)]
            except: return None
            edgeON = edges[edges[col]==True]
            edgeON.col = True
            edgeON = edgeON.rename(columns={'Time':'TimeOn'})
            edgeON = edgeON.drop(col,axis=1)
            edgeON.index = np.arange(len(edgeON))
            edgeOFF = edges[edges[col]==False]
            edgeOFF = edgeOFF.rename(columns={'Time':'TimeOff'})
            edgeOFF = edgeOFF.drop(col,axis=1)
            edgeOFF.index = np.arange(len(edgeOFF))
            test = pd.merge(edgeON,edgeOFF,left_index=True,right_index=True)
            test['dt'] = test.TimeOff-test.TimeOn
    
            for i, row in test.iterrows():
                start = int(np.where(df['Time'] == test['TimeOn'][i])[0])
                stop = int(np.where(df['Time'] == test['TimeOff'][i])[0])
        
                group.loc[group.index[range(start,stop)],'TasteID'] = taste
                delivery_idx.append(start)
                
        return group, delivery_idx
    
    new_df, delivery_idx = parse_edges(df, ['Line1', 'Line2'])
    
    def find_poke_dat(copy, poke, delivery_idx):
        # instantiate new columns with null values for later use
        copy['Taste_Delivery'] = False
        copy['Delivery_Time'] = None
        
        pokes = ['Time'] + [poke]
        data = copy[pokes]
        try: edges = data[data[poke].diff().fillna(False)]
        except: return None
        edgeON = edges[edges[poke]==True].shift(1)
        edgeON.iloc[0] = copy['Time'][0]
        edgeON['Poke2'].iloc[0] = True
        edgeON.col = True
        edgeON = edgeON.rename(columns={'Time':'TimeOn'})
        edgeON = edgeON.drop(poke,axis=1)
        edgeON.index = np.arange(len(edgeON))
        edgeOFF = edges[edges[poke]==False]
        edgeOFF = edgeOFF.rename(columns={'Time':'TimeOff'})
        edgeOFF = edgeOFF.drop(poke,axis=1)
        edgeOFF.index = np.arange(len(edgeOFF))
        test = pd.merge(edgeON,edgeOFF,left_index=True,right_index=True)
        test['dt'] = test.TimeOff-test.TimeOn
        
        delivery_time = []
        for i in delivery_idx:
            copy.loc[i,'Taste_Delivery'] = True
            copy.loc[i,'Delivery_Time'] = copy['Time'][i]
    
            # collect delivery time to erase Poke2 dat within 10 seconds of delivery
            delivery_time.append(copy['Time'][i])
    
        # generatees a new df with only delivery times (marked 'true')
        deliveries_only = copy.loc[copy['Taste_Delivery'] == True].reset_index(drop=True)
        
        second_copy = copy
        for i in delivery_time:
            second_copy = second_copy.loc[~((second_copy['Time'] > i) & (second_copy['Time'] < i+5)),:]
        
        for i, row in second_copy.iterrows():
            poke1 = np.where(second_copy['Taste_Delivery'] == True)[0]
            poke2 = poke1-1
        lat1 = second_copy['Time'].iloc[poke2].reset_index(drop=True)
        lat2 = second_copy['Time'].iloc[poke1].reset_index(drop=True)
        
        latencies = lat2.subtract(lat1) #look up how to subtract series from each other
        
        deliveries_only['Latencies'] = latencies
    
        return deliveries_only
    
    deliveries_only = find_poke_dat(new_df,'Poke2', delivery_idx)
    finaldf = finaldf.append(deliveries_only)
    
def add_days_elapsed(finaldf):
   
    new_df = finaldf
    
    res = []
    for name, group in new_df.groupby('AnID'):
        i=1
        for n, g in group.groupby('Date'):
            print(g)
            bit = np.zeros(len(g))
            bit = bit + i
            res.extend(bit)
            i += 1
    new_df['Sessions'] = res

    return new_df

def add_days_elapsed_again(finaldf):
   
    new_df = finaldf
    
    #res = []
    new_df['ones'] = 1
    
    tst= new_df[['AnID','Date','TasteID', 'Concentration']].drop_duplicates()
    tst = pd.pivot(tst, index = ['AnID','Date'], columns = 'TasteID', values='Concentration')
    tst['tasteset'] = tst['suc'] +'_&_'+ tst['qhcl']
    tst = tst.reset_index()
    new_df = new_df.merge(tst)    
    new_df['tastesession'] = new_df.groupby(['AnID','TasteID','tasteset'])['ones'].cumsum()
    # for name, group in new_df.groupby('AnID','Concentration'):
    #     i=1
    #     for n, g in group.groupby(['Date', 'Concentration']):
    #         print(g)
    #         bit = np.zeros(len(g))
    #         bit = bit + i
    #         res.extend(bit)
    #         i += 1

    return new_df

new_df = add_days_elapsed(finaldf)

def cumulativedels(new_df):
    csum = new_df.groupby(['AnID','Sessions','TasteID', 'Latencies']).Delivery_Time.sum()
    csum = csum.reset_index()
    return csum


csum = cumulativedels(new_df)
means = csum.groupby(["TasteID","Sessions"]).Delivery_Time.mean().reset_index()
fig, ax = plt.subplots(figsize=(10,5))
p1 = sns.scatterplot(data = csum, x = "Sessions", y = "Delivery_Time", hue = "TasteID", style = "AnID", s=65)
p2 = sns.lineplot(data = means, x = "Sessions", y = "Delivery_Time", hue = "TasteID")
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# finding the diff between suc and qhcl deliveries per day
copy = new_df

# create rates
copy2 = copy

# max_time = copy2.loc[copy['Time'].
# find the min/max time
max_time = copy2.groupby(['AnID', 'Sessions', 'Date']).max().reset_index()
minutes = max_time['Time']/60
# max_time['Minutes'] = max_time['Time']/60
max_time['Minutes'] = minutes

# test re-sums the deliveries without 'Section'
test = copy
test['Taste_Delivery'] = copy['Taste_Delivery'].astype(int)
test = copy.groupby(['AnID','Sessions', 'Date','TasteID']).agg(sum).reset_index()
test = test.drop(columns='Time')


max_time2 = max_time[['AnID', 'Date', 'Time']]
new_df2 = test.merge(max_time2, on=['AnID', 'Date'])

# max_time['Taste_Delivery'] = max_time['Taste_Delivery'].astype(int)
# deliveries = test['Taste_Delivery']
# deliveries = np.array(deliveries)

# max_time['Taste_Delivery'] = deliveries
# sns.relplot(data=max_time, x='Taste_Delivery', y='Minutes', hue='TasteID').set(title='rate of deliveries/minute across sessoins')

new_df2['Suc-Quin'] = new_df2.groupby(['AnID', 'Date', 'Sessions'])['Taste_Delivery'].diff().astype(float)
new_df2.loc[(test['Date'] < '091222') & (new_df2['TasteID'] == 'qhcl'), 'Concentration'] = 'qhcl_5mM'
new_df2.loc[(test['Date'] >= '091222') & (new_df2['TasteID'] == 'qhcl'), 'Concentration'] = 'qhcl_10mM'
new_df2.loc[new_df2['TasteID'] == 'suc', 'Concentration'] = 'suc_0.3M'

new_df3 = add_days_elapsed(new_df2)
new_df3 = add_days_elapsed_again(new_df2)
new_df3['deliveries_over_minutes'] = (new_df2['Taste_Delivery']/(new_df2['Time']/60))


del_rate_quin = new_df3.sort_values(['TasteID'], ascending=False)
del_rate_quin = del_rate_quin.sort_values(['Date', 'AnID', 'Sessions'])
del_rate_quin['delivery_rate_diff'] = del_rate_quin.groupby(['AnID', 'Date', 'Sessions'])['deliveries_over_minutes'].diff()
del_rate_quin['delivery_rate_diff'] = del_rate_quin['delivery_rate_diff'] * -1


t = sns.barplot(data = del_rate_quin, x='tasteset', y='delivery_rate_diff')
t.set(title='sucrose preference increases with concentration of qhcl', ylabel='suc - qhcl (rewards/min)', xlabel='', xticklabels=["5mM qhcl & 0.3M suc", "10mM qhcl & 0.3M suc"])

save_dir = '/Users/emmabarash/lab/sfn_figs/'
fig = t.get_figure()
t.figure.savefig(save_dir+'rate_summary.svg', bbox_inches='tight')

# sns.set_theme(style='white', font_scale=1.25)
g = sns.lineplot(data = del_rate_quin.loc[(del_rate_quin['delivery_rate_diff'] != 'nan')],
            x = 'tastesession', y = 'delivery_rate_diff', hue='tasteset')
g.axhline(0, color='black', linestyle='dashed')
g.set(title='Sucrose Preference (deliveries/min), N=2', ylabel='suc - qhcl (rewards/min)',xlabel='Sessions')
plt.legend(loc='lower right')
# g.fit.set_size_inches()

save_dir = '/Users/emmabarash/lab/sfn_figs/'
fig = g.get_figure()
fig.savefig(save_dir+'rate_fig.svg', bbox_inches='tight')

not_null_set = del_rate_quin.loc[(del_rate_quin['delivery_rate_diff'].notnull())]
a = del_rate_quin['delivery_rate_diff'].loc[del_rate_quin['tasteset'] == 'suc_0.3M_&_qhcl_5mM']
b = del_rate_quin['delivery_rate_diff'].loc[del_rate_quin['tasteset'] == 'suc_0.3M_&_qhcl_10mM']
[stats,p_val] = scipy.stats.ttest_ind(a,b)


g = sns.lineplot(data = del_rate_quin.loc[(del_rate_quin['delivery_rate_diff'] != 'nan')],
            x = 'Sessions', y = 'delivery_rate_diff', hue='Concentration')
g.axhline(0, color='black', linestyle='dashed')
g.set(title='Sucrose Preference, N=2', ylabel='suc rate - qhcl rate')

g = sns.relplot(data=new_df2, kind='line', x='Sessions', y='deliveries_over_minutes', hue='Concentration')

# plot mean diff for both animals
g = sns.lineplot(data = new_df2.loc[(new_df2['Suc-Quin'] != 'nan')],
            x = 'Sessions', y = 'Suc-Quin', hue='Suc-Quin')
g.axhline(0, color='black', linestyle='dashed')
g.set(title='Difference suc-qhcl Across Sessions')

#same as above but lineplot
g = sns.lineplot(data = new_df2.loc[(new_df2['Suc-Quin'] != 'nan')],
            x = 'Sessions', y = 'Suc-Quin', hue='Concentration')
g.axhline(0, color='black', linestyle='dashed')
g.set(title='Difference suc-qhcl Across Sessions, N=2')

#same as above but lineplot
plt.xticks(rotation = 90)
g = sns.lineplot(data = test.loc[(test['Suc-Quin'] != 'nan')],
            x = 'Date', y = 'Suc-Quin')
g.axhline(0, color='black', linestyle='dashed')
g.set(title='Difference suc-qhcl Across Sessions, N=2')

# plot animals individually
g = sns.lineplot(data = test.loc[(test['Suc-Quin'] != 'nan')],
            x = 'Sessions', y = 'Suc-Quin', hue = 'AnID')

g = sns.catplot(data = test.loc[(test['Suc-Quin'] != 'nan')],
                kind = 'bar', x = 'AnID', y = 'Suc-Quin', hue = 'AnID')

# diff over sum would be -1 and +1
# avg ^ that over animals
g = sns.displot(copy, x='Latencies')
