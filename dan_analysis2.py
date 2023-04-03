#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:40:52 2022

@author: danielsvedberg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


anID = ['EB1', 'EB2', 'EB3', 'EB4', 'EB5', 'EB6','EB7', 'EB8', 'EB9']
dshape = [19, 27, 27, 9, 9, 9, 9, 9, 9]
dlearn = [0, 0, 0, 6, 4, 4, 5, 0, 4]
training = ['subtractive','subtractive','subtractive','additive','additive',
            'additive','additive','additive','additive']

data = pd.DataFrame(list(zip(anID,training,dshape,dlearn)), columns = ['anID','training','days_shaping','days_learning'])
data['days_to_criterion'] = data['days_shaping']+data['days_learning']




sns.set(style = 'ticks', font_scale=1.5, rc={'figure.figsize':(10,15)})
g = sns.catplot(x="training", y = "days_to_criterion",data = data,
                kind = 'bar',capsize = 0.2, facecolor=(1, 1, 1, 0),edgecolor=".2")
sns.swarmplot(x="training", y = "days_to_criterion", data=data,
              color="black", size = 10, ax = g.ax)
g.set(ylabel = "# sessions to criterion")
plt.savefig('d2c.png',dpi=300)


#t-test https://www.marsja.se/how-to-perform-a-two-sample-t-test-with-python-3-different-methods/
sub = data.query('training == "subtractive"')['days_to_criterion']
add = data.query('training == "additive"')['days_to_criterion']
import scipy.stats as stats
stats.levene(sub,add) #test to see if variances are equal--pvalue is over 0.05 so it passes
res = stats.ttest_ind(sub,add,equal_var = True) #actual t-test done here
display(res) #displays T-statistic and p-value, pvalue=0.001! add ** for significance in powerpoint

