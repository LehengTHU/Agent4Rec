import numpy as np
import pandas as pd
import os
import pickle
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from mpl_toolkits.mplot3d.axes3d import Axes3D # 3D engine
import scipy.stats as stats

def helper_load_train(filename):
    user_dict_list = {}
    item_dict = set()
    item_dict_list = {}
    trainUser, trainItem = [], []

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            # print(line)
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            # LightGCN
            trainUser.extend([user] * len(items))
            trainItem.extend(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

            for item in items:
                if item in item_dict_list.keys():
                    item_dict_list[item].append(user)
                else:
                    item_dict_list[item] = [user]

    return user_dict_list, item_dict, item_dict_list, trainUser, trainItem

def save_user_dict_to_txt(user_dict, base_path, filename):
    with open(base_path + filename, 'w') as f:
        for u, v in user_dict.items():
            f.write(str(int(u)))
            for i in v:
                f.write(' ' + str(int(i)))
            f.write('\n')

def fix_seeds(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def prepare_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        os.system("rm -rf " + dir_path)
        os.makedirs(dir_path)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def hist_group_results(groups, bins=20):
    for i, g in enumerate(groups):
        plt.hist(g, bins=bins, alpha=0.4, label=i+1)
    plt.legend()
    plt.show()

# def hist_group_results(groups, bins=20):
#     for g in groups:
#         plt.hist(g, bins=bins, alpha=0.4)
#     plt.legend()
#     plt.show()

def bar_group_mean(n_groups, groups):
    x_list = [str(i) for i in range(1, n_groups+1)]
    print(groups[0])
    print(x_list)
    # Use seaBorn to draw a bar chart.
    # sns.set_style("whitegrid")
    # sns.set_context("paper")
    # sns.set(font_scale=1.5)
    plt.figure(figsize=(12, 3))

    plt.subplot(1, 3, 1)
    sns.barplot(x=x_list, y=groups[0], palette="Blues_d")
    plt.xlabel("Activity")

    plt.subplot(1, 3, 2)
    sns.barplot(x=x_list, y=groups[1], palette="Blues_d")
    plt.xlabel("Conformity")

    plt.subplots_adjust(left=0.001, right=0.999, top=0.9, bottom=0.1)
    plt.subplot(1, 3, 3)
    sns.barplot(x=x_list, y=groups[2], palette="Blues_d")
    plt.xlabel("Diversity")

    # Restore settings
    # sns.set()

def bar_rating(x, y, y_label):
    # colors = ["#668CAD", "#E7AC72", "#174F75", "#575C75"]
    f_size = 28
    spine_linewidth = 1.4
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(spine_linewidth)
    ax.spines['left'].set_linewidth(spine_linewidth)
    ax.spines['top'].set_linewidth(spine_linewidth)
    ax.spines['right'].set_linewidth(spine_linewidth)
    plt.ylabel(y_label, fontsize=f_size, fontweight ='bold')
    plt.xlabel('User Rating', fontsize=f_size, fontweight ='bold')
    plt.tick_params(labelsize=int(0.7*f_size))
    plt.bar(x, y, width=1, alpha=0.8, color="#174F75")
    # plt.plot(g_rating['rating'], g_rating['counts'], marker='s', markersize=10, color=colors[2])
    # legend_font = {
    #     'size': int(f_size*0.6),
    # 'weight': "bold",  # Whether to bold or not, not bold
    # }
    # ax.legend(prop=legend_font, loc='upper right')
    plt.show()

def plot_rating(x, y, type_name):
    # colors = ["#668CAD", "#E7AC72", "#174F75", "#575C75"]
    f_size = 44
    spine_linewidth = 3

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(spine_linewidth)
    ax.spines['left'].set_linewidth(spine_linewidth)
    ax.spines['top'].set_linewidth(spine_linewidth)
    ax.spines['right'].set_linewidth(spine_linewidth)

    plt.ylabel("Ratio", fontsize=f_size, fontweight ='bold')
    plt.xlabel('Rating', fontsize=f_size, fontweight ='bold')
    plt.tick_params(labelsize=int(0.7*f_size))
    plt.xticks(weight='bold')
    plt.yticks(weight='bold')
    # plt.plot(x, y, marker='s', markersize=10, color="#174F75")
    if(type_name == 'true'):
        ax.plot([1, 2, 3, 4, 5], y, marker='s', markersize=15, markeredgewidth=2, color="#9b1e15",  markeredgecolor='#9b1e15', markerfacecolor='#FEF9F5', linewidth=3)
        # Set the y-axis scale to increments of 0.1.
        plt.yticks([0.1, 0.2, 0.3])

        plt.savefig('../assets/alignment/rating_true.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    else:
        ax.plot([1, 2, 3, 4, 5], y, marker='s', markersize=16, markeredgewidth=2, color="#174F75", markeredgecolor='#174F75', markerfacecolor='#F9FBFD', linewidth=3)
        plt.yticks([0, 0.2, 0.4])
        plt.savefig('../assets/alignment/rating_sim.png', dpi=300, bbox_inches='tight', pad_inches=0.05)

    plt.show()


def pair_plot(y1, y2):
    # Plot y1 and y2 on the same graph, using two subplots.
    f_size = 44
    spine_linewidth = 4

    fig = plt.figure(figsize=(22, 7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)


    # Set background color.
    ax1.spines['bottom'].set_linewidth(spine_linewidth)
    ax1.spines['left'].set_linewidth(spine_linewidth)
    ax1.spines['top'].set_linewidth(spine_linewidth)
    ax1.spines['right'].set_linewidth(spine_linewidth)
    ax1.tick_params(labelsize=int(0.8*f_size))
    ax1.set_ylabel('Ratio', fontsize=f_size, fontweight ='bold')
    ax1.plot([1, 2, 3, 4, 5], y1, marker='s', markersize=15, markeredgewidth=2, color="#9b1e15",  markeredgecolor='#9b1e15', markerfacecolor='#FEF9F5', linewidth=3)
    # Set the name of the y-axis.

    ax2.spines['bottom'].set_linewidth(spine_linewidth)
    ax2.spines['left'].set_linewidth(spine_linewidth)
    ax2.spines['top'].set_linewidth(spine_linewidth)
    ax2.spines['right'].set_linewidth(spine_linewidth)
    ax2.tick_params(labelsize=int(0.8*f_size))
    ax2.set_ylabel('Ratio', fontsize=f_size, fontweight ='bold')
    ax2.plot([1, 2, 3, 4, 5], y2, marker='s', markersize=16, markeredgewidth=2, color="#174F75", markeredgecolor='#174F75', markerfacecolor='#F9FBFD', linewidth=3)
    # Set the font size of the tick labels for two subplots.

    # Set the font.
    plt.rcParams['font.sans-serif']=['Times New Roman']

    # Set the x-axis.
    ax1.set_xlabel('User Rating', fontsize=f_size, fontweight ='bold')
    ax2.set_xlabel('Agent Rating', fontsize=f_size, fontweight ='bold')
    # Set the data for the x-axis ticks.
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax2.set_xticks([1, 2, 3, 4, 5])
    # for ax in fig.axes:
    #     ax.tick_params(axis='both', which='major', labelsize=int(0.8*f_size), width=3, length=10, pad=10, labelweight='bold')
    for ax in fig.axes:
        ax.tick_params(axis='both', which='major', labelsize=int(0.8*f_size), width=3, length=10, pad=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_weight("bold")
    # Set the spacing between two subplots.
    plt.subplots_adjust(wspace=0.22)
    plt.savefig('../assets/alignment/rating_alignment.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()


def plot_bar(data):
    f_size = 28
    spine_linewidth = 1.4
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(spine_linewidth)
    ax.spines['left'].set_linewidth(spine_linewidth)
    ax.spines['top'].set_linewidth(spine_linewidth)
    ax.spines['right'].set_linewidth(spine_linewidth)
    # plt.ylabel(y_label, fontsize=f_size, fontweight ='bold')
    # plt.xlabel('Rating', fontsize=f_size, fontweight ='bold')
    plt.tick_params(labelsize=int(0.7*f_size))
    plt.bar(range(len(data)), data, width=1, alpha=0.8, color="#174F75")
    plt.show()


def hist_3D(group_data, labels, is_discrete=True, bins=60):
    colors = ["# 83CDD8","#668CAD",  "#174F75", "#575C75"] # Blue
    f_size = 28
    fig = plt.figure(figsize=(9,6))
    ax3 = Axes3D(fig)
    fig.add_axes(ax3)
    for i, g in enumerate(group_data):
        if(is_discrete):
            counts = Counter(g)
            ax3.bar(counts.keys(), counts.values(),
                    zs = i+1,
                    zdir = 'y',# In which direction, arranged in rows and rows.
                    label = labels[i],
                    alpha = 0.7,# alpha transparency
                    width = 1, 
                    color=colors[i])
        else:
            g = [d for d in g if d < 1.0]
            hist, edges = np.histogram(g, bins=bins, density=False)
            width = edges[1] - edges[0]
            center = (edges[:-1] + edges[1:]) / 2
            ax3.bar(center, hist,
                    zs = i+1,
                    zdir = 'y',# In which direction, arranged in rows and rows.
                    label = labels[i],
                    alpha = 0.7,# alpha transparency
                    width = width, 
                    align='center',
                    color=colors[i])
            
    # Bold coordinate axis
    spine_linewidth = 1.4
    ax3.spines['bottom'].set_linewidth(spine_linewidth)
    ax3.spines['left'].set_linewidth(spine_linewidth)
    ax3.spines['top'].set_linewidth(spine_linewidth)
    ax3.spines['right'].set_linewidth(spine_linewidth)
    # Set the scale of the Y-axis.
    ax3.set_yticks(np.arange(1,4))
    # Set the range of the Y-axis.
    ax3.set_ylim(0,4)
    ax3.set_xlabel('Interaction Numbser',fontsize = 18, fontweight ='bold')
    ax3.set_ylabel('Group',fontsize = 18, fontweight ='bold')
    ax3.set_zlabel('Ratio',fontsize = 18, fontweight ='bold')

    legend_font = {
        'size': int(f_size*0.6),
        'weight': "bold",  # Whether to bold or not to bold.
    }
    locs, _ = plt.yticks() 
    # print(locs)
    total_samples = sum([len(g) for g in group_data])
    # plt.yticks(locs, ["{:.2f}".format(i) for i in np.round(locs/total_samples,2)])
    # Set Y-axis tick labels.
    ax3.set_yticklabels(["{:.2f}".format(i) for i in np.round(locs/total_samples,2)])
    ax3.legend(prop=legend_font, loc='upper right')
    plt.show()

def significance_test(group_data):
    f_statistic, p_value = eval("stats.f_oneway(" + ",".join(["group_data[{}]".format(i) for i in range(len(group_data))]) + ")")
    print(f"F-statistic: {f_statistic}")
    print(f"P-value: {p_value}\n")
    return f_statistic, p_value

def hist_2D(group_data, labels, save_name, x_label, order, text_place, is_discrete=True, bins=60):
    colors = ["# 83CDD8","#668CAD",  "#174F75", "#575C75"] # Blue
    f_size = 44
    spine_linewidth = 3
    f_statistic, p_value = significance_test(group_data)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(spine_linewidth)
    ax.spines['left'].set_linewidth(spine_linewidth)
    ax.spines['top'].set_linewidth(spine_linewidth)
    ax.spines['right'].set_linewidth(spine_linewidth)
    plt.ylabel('Proportion of Population', fontsize=f_size, fontweight ='bold')
    plt.xlabel(x_label, fontsize=f_size, fontweight ='bold')
    plt.tick_params(labelsize=int(0.7*f_size))
    max_value = 0
    # for i, g in enumerate(group_data):
    for i in order:
        g = group_data[i]
        if(is_discrete):
            counts = Counter(g)
            max_value = max(max_value, max(counts.values()))
            ax.bar(counts.keys(), counts.values(), width=1, alpha=0.8, label=labels[i], color=colors[i])
        else:
            hist, edges = np.histogram(g, bins=bins, density=False)
            width = edges[1] - edges[0]
            center = (edges[:-1] + edges[1:]) / 2
            max_value = max(max_value, max(hist))
            ax.bar(center, hist, width=width, align='center', alpha=0.8, label=labels[i], color=colors[i])

    plt.ylim(0, 1.3*max_value)
    legend_font = {
        'size': int(f_size*0.6),
        'weight': "bold",  # Whether to bold or not to bold.
    }
    locs, _ = plt.yticks() 
    # print(locs)
    total_samples = sum([len(g) for g in group_data])
    plt.yticks(locs, ["{:.2f}".format(i) for i in np.round(locs/total_samples,2)], weight='bold')
    
    plt.xticks(weight='bold')

    handles, labels_ = ax.get_legend_handles_labels()
    # print(handles, labels_)
    ax.legend([handles[i] for i in order], [labels_[i] for i in order], prop=legend_font, loc='best')
    
    # plt.rcParams['ytick.weight'] = 'bold'
    # plt.rcParams['xtick.weight'] = 'bold'

    # If the p-value has more than four decimal places, it should be expressed in scientific notation.
    if p_value < 0.01:
        p_value = "{:.2e}".format(p_value)
    else:
        p_value = "{:.2f}".format(p_value)
    # Set the range of the y-axis.
    # plt.title('p-value: {}'.format(p_value), fontsize=f_size, fontweight ='bold', pad=15)
    # fig.tight_layout()
    # Add text in appropriate places in the image to avoid obstruction.
    plt.text(text_place[0], text_place[1], 'p-value: {}'.format(p_value), fontsize=int(f_size*0.6), fontweight ='bold', ha='center', va='center')
    plt.savefig(f'../assets/alignment/{save_name}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.show()

