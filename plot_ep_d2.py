# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:11:01 2023

@author: Student
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:21:44 2023

@author: Student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def get_plot(data, savefig=False, log=False):
    dat = data
    
    if log:
        dat['MSE'] = np.log(dat['MSE'].to_numpy())
        dat['Stderr'] = np.log(dat['Stderr'].to_numpy())
        dat['Value'] = np.log(dat['Value'].to_numpy())
    else:
        dat['MSE'] = dat['MSE'].to_numpy()
        dat['Stderr'] = dat['Stderr'].to_numpy()
        dat['Value'] = dat['Value'].to_numpy()
    
    #print(dat['MSE'], dat['Stderr'])
    
    fig, ax1 = plt.subplots(figsize=(10,6))

    sns.set(style= "whitegrid")
    #colors = ['magenta', 'green', 'blue', 'orange','red']
    marker=['-','--',':']
    # colors = ['cyan', 'blue', 'orange']
    # iterColors = iter(colors)
    lvls = dat.Group.unique()
    m_i = 0
    for i in lvls:
        ax1.errorbar(x = dat[dat["Group"]==i]["d"],
                    y=dat[dat["Group"]==i]["MSE"],
                    yerr=dat[dat["Group"]==i]["Stderr"], label=i,
                    ls=marker[m_i],c="#377eb8",
                    ecolor='#377eb8')
                    #ecolor = next(iterColors))
        m_i += 1
    # for i,j in enumerate(ax1.lines):
    #     j.set_color(colors[i])
    plt.xlabel("Dimension of Data (d)", fontsize=20)
    plt.ylabel("Mean Squared Error(MSE)", fontsize=20)
    
    ax1.set_ylim([0, 0.00014])
    #ax1.set_ylim([0, 0.002])
    
    plt.title("Bayes Error Rate(BER) ", fontsize=25)
    #plt.title("Rényi Integral of order alpha", fontsize=25)
    ax1.yaxis.label.set_color('#377eb8')
    ax1.xaxis.label.set_color('#d95f02')
    ax1.tick_params(axis='x', colors='#d95f02')
    ax1.tick_params(axis='y', colors='#377eb8')
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20,loc='upper left');
    
    #sns.set(style= "whitegrid")
    #colors1 = ['red', 'black']
    ls1=['--', ':']
    #iterColors1 = iter(colors1)
    ax2=ax1.twinx()
    lvls1 = dat.Group1.unique()[:2]
    
    
    m_j = 0
    for l in lvls1:
        ax2.plot( dat[dat["Group1"]==l]["d1"],
                    dat[dat["Group1"]==l]["Value"],label=l, 
                    c="#1b9e77",ls=ls1[m_j])
        m_j=+1
    # for l,k in enumerate(ax2.lines):
    #     k.set_color(colors1[l]) 
    #plt.xlabel("Sample Size(n)", fontsize=18)
    plt.legend(fontsize=20,loc='best');
    plt.yticks(fontsize=20)
    # ax2.set_ylabel("Optimum of Epcelon", fontsize=10)
    ax2.set_ylim([0, 0.45])
    #plt.title("Ensemble versus Fixed Kernel", fontsize=20)
    #plt.xticks(fontsize=18)
    #ax2.set_yticks(fontsize=10)
    #plt.legend(fontsize=14);
    ax2.yaxis.label.set_color('#1b9e77')
    ax2.tick_params(axis='y', colors='#1b9e77')
        
    if savefig:
        save_file = 'ensemble_' + str(log) + '.png'
        save_file_dir = './'
        plt.savefig(save_file_dir + save_file)
    
    fig.tight_layout()
    plt.show()
    

    
dat = pd.read_excel('C:/Users/Student/Desktop/dd22_non_trunc_BER.xlsx', engine='openpyxl')
#dat = pd.read_excel('C:/Users/Student/Desktop/dd22_trunc_MG.xlsx', engine='openpyxl')
get_plot(dat, savefig=True, log=False)