import matplotlib.pyplot as plt
import numpy as np

auc = {}
los = {}

f_xDeepFM = open('./xDeepFM/log1.txt','r').readlines()[166:17040]
auc['xDeepFM'] = [float(line.split('-')[3][6:12]) for line in f_xDeepFM]
los['xDeepFM'] = [float(line.split('-')[2][7:11]) for line in f_xDeepFM]
# print(auc['xDeepFM'])

f_WDL = open('./WDL/log1.txt','r').readlines()[163:17036]
auc['WDL'] = [float(line.split('-')[3][6:12]) for line in f_WDL]
los['WDL'] = [float(line.split('-')[2][7:11]) for line in f_WDL]
# print(auc['WDL'])

f_PNN = open('./PNN/log1.txt','r').readlines()[222:17095]
auc['PNN'] = [float(line.split('-')[3][6:12]) for line in f_PNN]
los['PNN'] = [float(line.split('-')[2][7:11]) for line in f_PNN]
# print(auc['PNN'])

f_NFM = open('./NFM/log1.txt','r').readlines()[173:12226]
auc['NFM'] = [float(line.split('-')[3][6:12]) for line in f_NFM]
los['NFM'] = [float(line.split('-')[2][7:11]) for line in f_NFM]
# print(auc['NFM'])

f_DeepFM = open('./DeepFM/log1.txt','r').readlines()[161:1214]
auc['DeepFM'] = [float(line.split('-')[3][6:12]) for line in f_DeepFM]
los['DeepFM'] = [float(line.split('-')[2][7:11]) for line in f_DeepFM]
# print(auc['DeepFM'])

flag = 1000

with plt.style.context("bmh"):
    font = {"color": "darkred",
            "size":10,
            "family" : "times new roman"}
    fig, axs = plt.subplots(1,4,figsize=(12,2.5))

    axs0 = axs[0].twinx()
    lb1 = axs[0].plot(auc['xDeepFM'][:flag],'r--',label='AUC', linewidth=1.5)
    lb2 = axs0.plot(los['xDeepFM'][:flag],'g--',label='log loss', linewidth=1.5)
    axs[0].set_xlabel('each batch', fontdict=font)
    axs[0].set_ylabel('AUC', fontdict=font)
    axs0.set_ylabel('log loss', fontdict=font)
    axs[0].set_title('xDeepFM', fontdict=font)
    axs[0].legend(lb1 + lb2, [l.get_label() for l in lb1 + lb2], loc=0, bbox_to_anchor=(1,0.5), prop={'family': 'Times New Roman', 'weight': 'normal', 'size': 10})
    for ax in axs:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        ax.tick_params(labelsize=10)

    for label in axs0.get_xticklabels() + axs0.get_yticklabels():
        label.set_fontname('Times New Roman')
    axs0.tick_params(labelsize=10)


    axs1 = axs[1].twinx()
    lb1 = axs[1].plot(auc['WDL'][:flag],'r--',label='AUC', linewidth=1.5)
    lb2 = axs1.plot(los['WDL'][:flag],'g--',label='log loss', linewidth=1.5)
    axs[1].set_xlabel('each batch', fontdict=font)
    axs[1].set_ylabel('AUC', fontdict=font)
    axs1.set_ylabel('log loss', fontdict=font)
    axs[1].set_title('WDL', fontdict=font)
    axs[1].legend(lb1 + lb2, [l.get_label() for l in lb1 + lb2], loc=0, bbox_to_anchor=(1,0.5), prop={'family': 'Times New Roman', 'weight': 'normal', 'size': 10})
    for ax in axs:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        ax.tick_params(labelsize=10)
    for label in axs1.get_xticklabels() + axs1.get_yticklabels():
        label.set_fontname('Times New Roman')
    axs1.tick_params(labelsize=10)

    axs2 = axs[2].twinx()
    lb1 = axs[2].plot(auc['DeepFM'][:flag],'r--',label='AUC', linewidth=1.5)
    lb2 = axs2.plot(los['DeepFM'][:flag],'g--',label='log loss', linewidth=1.5)
    axs[2].set_xlabel('each batch', fontdict=font)
    axs[2].set_ylabel('AUC', fontdict=font)
    axs2.set_ylabel('log loss', fontdict=font)
    axs[2].set_title('DeepFM', fontdict=font)
    axs[2].legend(lb1 + lb2, [l.get_label() for l in lb1 + lb2], loc=0, bbox_to_anchor=(1,0.5), prop={'family': 'Times New Roman', 'weight': 'normal', 'size': 10})
    for ax in axs:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        ax.tick_params(labelsize=10)
    for label in axs2.get_xticklabels() + axs2.get_yticklabels():
        label.set_fontname('Times New Roman')
    axs2.tick_params(labelsize=10)


    axs3 = axs[3].twinx()
    lb1 = axs[3].plot(auc['PNN'][:flag],'r--',label='AUC', linewidth=1.5)
    lb2 = axs3.plot(los['PNN'][:flag],'g--',label='log loss', linewidth=1.5)
    axs[3].set_xlabel('each batch', fontdict=font)
    axs[3].set_ylabel('AUC', fontdict=font)
    axs3.set_ylabel('log loss', fontdict=font)
    axs[3].set_title('PNN', fontdict=font)
    axs[3].legend(lb1 + lb2, [l.get_label() for l in lb1 + lb2], loc=0, bbox_to_anchor=(1,0.5), prop={'family': 'Times New Roman', 'weight': 'normal', 'size': 10})
    for ax in axs:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')
        ax.tick_params(labelsize=10)
    for label in axs3.get_xticklabels() + axs3.get_yticklabels():
        label.set_fontname('Times New Roman')
    axs3.tick_params(labelsize=10)


    plt.tight_layout()
    plt.show()

