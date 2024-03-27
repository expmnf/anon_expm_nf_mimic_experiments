# plot_results.py

# %%
import sys, copy 
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(Path(".", "code").absolute().as_posix())
from data_utils import *
sys.path.append(Path(".", "code", 'mimic_experiment').absolute().as_posix())
from config import * #experiment configuration variables & results path
import warnings, logging
warnings.simplefilter(action='ignore', category=FutureWarning) #supress future warnings
warnings.simplefilter(action='ignore', category=UserWarning) #supress user warnings
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
snot = '#00D91A' # dpsgd color

# %%
# data folders
dpsgd_folder = Path(RESULTS_FOLDER, 'grud_dpsgd', 'final0')
nonpriv_folder = Path(RESULTS_FOLDER, 'grud_nonprivate_baseline', 'final0')
expm_folder = Path(RESULTS_FOLDER, 'grud_expm', 'final0')

# read in data
df_nonpriv = pd.DataFrame.from_records(
    map(unjsonify, nonpriv_folder.glob('**/*.json')))[['target', 'loss',  'use_bn', 'seed', 'auc']]
baseline = df_nonpriv.groupby(['target', 'loss', 'use_bn']).median(numeric_only=True)['auc'] # Series w/ median AUC for each task
# baseline.loc[('los_3', 'l2', 'bn')] ## gives the median auc for los_3, l2 loss, bn
add_baseline = True # indictor for if to plot baseline or not. 

df_dpsgd = pd.DataFrame.from_records( map( unjsonify, dpsgd_folder.rglob('*.json')))[['target', 'loss', 'target_epsilon',  'auc', 'eps_error']].rename(columns = {'auc': "aucs"})

df_expm = pd.DataFrame.from_records(
    map(unjsonify, expm_folder.rglob('*.json')))[['target', 'epsilon', 'aucs', 'seed']].explode('aucs')

# print(sorted(df_dpsgd.target_epsilon.unique()))


# %% Make table baseline bn v no-bn and 
for task in ['los_3', 'mort_icu']:
    df = pd.DataFrame(baseline).reset_index()
    df = df[df.target == task][df.columns.difference(['target'])]    
    df = df.pivot(index = 'loss', columns = 'use_bn')
    df = df.T.style.format(precision=4) # sets number of decimals
    print(task)
    print(df.to_latex())
    print()


# %% 
# make folder for plots: 
for task in ['los_3', 'mort_icu']:
    Path(GRUD_PLOTS_PATH, task).mkdir(exist_ok=True)

# %% 
# epsilon 0.5 - 10:
m, M= .5, 10 
df_epre = df_expm[(df_expm.epsilon >= m) & (df_expm.epsilon <= M) ].sort_values('epsilon')
df_dpre = df_dpsgd[(df_dpsgd.target_epsilon >= m) & (df_dpsgd.target_epsilon <= M) ].sort_values('target_epsilon')

for task in ['los_3', 'mort_icu']:
    df_e = df_epre[df_epre.target == task]
    df_d = df_dpre[df_dpre.target == task]
    # makes expm plot w/ median markers
    fig, ax = plt.subplots(layout = "constrained", figsize = (6.5, 5))
    sns.pointplot(data = df_e, ax = ax, x = "epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = 'purple', 
                markers = 'D',  label = "ExpM+NF")#native_scale = True, 
    # makes dpsgd plots
    sns.pointplot(data = df_d[df_d.loss == 'l2'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = snot, 
                markers = 'D',  label = r'DPSGD, $\ell(2)$ loss $\delta=1E-5$')
    sns.pointplot(data = df_d[df_d.loss == 'bce'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = 'green', scale = 1.25, 
                markers = 'o',  label = r'DPSGD, BCE Loss, $\delta=1E-5$')

    ax.set_xlabel(r'Privacy ($\epsilon$)', fontsize = 16)
    ax.set_ylabel( 'Accuracy (AUC)', fontsize = 16)
    if add_baseline: 
        ax.axhline(y=baseline[task].max(), color="k", linestyle="--", linewidth = 1, 
#                    label = f'baseline (no privacy), {baseline[task].max():0.4}')
                    label = f'GRUD (no privacy), {baseline[task].max():0.4}')
        # ax.axhline(y=baseline[(task, "l2", "nobn")], color="k", linestyle="--", linewidth = 1, 
        #             label = f'baseline (no privacy), {baseline[task, "l2", "nobn"]:.4}')
    if task == 'los_3':
        # plt.title(f"MIMIC3 Length of Stay GRU-D ExpM+NF & DPSGD Median AUC Results")
        if add_baseline: 
            ym, yM = plt.ylim()
            plt.ylim(.675, yM)

    else: ## task == 'mort_icu': 
    #     plt.title((f"MIMIC3 Mortality GRUD-D ExpM+NF & DPSGD Median AUC Results"))
        if add_baseline: 
            ym, yM = plt.ylim()
            plt.ylim(.75, yM)
    
    if add_baseline: 
        leg = ax.legend()
        leg_lines = leg.get_lines() # make legend baseline dotted
        leg_lines[-1].set_linestyle(":")
    # change the line width for the legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [copy.copy(ha) for ha in handles ]
    plt.legend(handles=handles, labels=labels, loc = 'lower right')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    print(task)
    fig.savefig(Path(GRUD_PLOTS_PATH, task,  'epsilon_2e-1_to_10.png'), bbox_inches='tight')
    plt.show()
    print()

# %% 
## Now epsilon .0001, 1, log plot. 
m, M= .0001, 1 
df_epre = df_expm[(df_expm.epsilon >= m) & (df_expm.epsilon <= M) & (df_expm.epsilon != .5)].sort_values('epsilon')
df_dpre = df_dpsgd[(df_dpsgd.target_epsilon >= m) & (df_dpsgd.target_epsilon <= M) & (df_dpsgd.target_epsilon != .5) ].sort_values('target_epsilon')

for task in ['los_3', 'mort_icu']:
    df_e = df_epre[df_epre.target == task]
    df_d = df_dpre[df_dpre.target == task]

    # makes expm plot w/ median markers
    fig, ax = plt.subplots(layout = "constrained", figsize = (6.5, 5))
    
    sns.pointplot(data = df_e, ax = ax, x = "epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = 'purple', 
                markers = 'D', label = "ExpM+NF", native_scale = True)
    # makes dpsgd plots
    sns.pointplot(data = df_d[df_d.loss == 'l2'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = snot, native_scale = True, 
                markers = 'D', label = r'DPSGD. $\ell(2)$ loss, $\delta=1E-5$')
    sns.pointplot(data = df_d[df_d.loss == 'bce'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = 'green', native_scale = True, scale = 1.25,
                markers = 'o', label = r'DPSGD, BCE loss, $\delta=1E-5$')
    
    if add_baseline: ax.axhline(y=baseline[task].max(), color="k", 
                    linestyle="--", linewidth = 1, 
#                    label = f'baseline (no privacy), {baseline[task].max():.4}')
                    label = f'GRUD (no privacy), {baseline[task].max():.4}')
    
    ax.set_xscale('log')
    ax.set_xlabel(r'Privacy ($\epsilon$), log scale', fontsize = 16)
    ax.set_ylabel( 'Accuracy (AUC)', fontsize = 16)

    if task == 'los_3':
        if add_baseline: 
            ym, yM = plt.ylim()
            plt.ylim(.46, .75)

    else: ## task == 'mort_icu': 
    #     plt.title((f"MIMIC3 Mortality GRUD-D ExpM+NF & DPSGD Median AUC Results"))
        if add_baseline: 
            ym, yM = plt.ylim()
            plt.ylim(ym, yM)

    if add_baseline: 
        leg = ax.legend()
        leg_lines = leg.get_lines()
        leg_lines[-1].set_linestyle(":")
    # change the line width for the legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [copy.copy(ha) for ha in handles ]
    plt.legend(handles=handles, labels=labels, loc = 'lower right')
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    print(task)
    fig.savefig(Path(GRUD_PLOTS_PATH, task, 'epsilon_1e-4_to_1.png'), bbox_inches='tight')
    plt.show()
    print()

# %% box plots 
df_epre = df_expm[df_expm.epsilon.isin([.0001, .001, .01, .1, 1, 10])].sort_values('epsilon')

for task in ['los_3', 'mort_icu']:
    df_e = df_epre[df_epre.target == task]
    df_e.aucs = pd.to_numeric(df_e.aucs)
    df_eg = df_e.groupby(['epsilon', 'seed']).mean(numeric_only=True).reset_index()

    ## makes expm box plots
    sns.catplot(data = df_e, x = "epsilon", y = "aucs", kind = 'box', 
                    showmeans=False, showfliers=False, width=0.37,  height=5, aspect=6.5/5,
                    medianprops={"color": "purple", "linewidth": 2},
                    boxprops={"facecolor": "white", "edgecolor": "purple",
                            "linewidth": 0.5, 'alpha' : .5},
                    whiskerprops={"color": "purple", "linewidth": 2, 'alpha': .5},
                    capprops={"color": "purple", "linewidth": 1.5, 'alpha':.5}, 
                    native_scale=True, legend='auto', log_scale = (True ,False))
    ## adds the 10 models' means to show that some runs did poorly and pulled down the distribution 
    sns.swarmplot(data = df_eg,  x = "epsilon", y = "aucs", 
                native_scale=True, alpha = .5, legend = True,
                color ='purple', size = 3, log_scale = (True ,False)) 

    plt.xlabel(r'Privacy ($\epsilon$), log scale', fontsize = 16)
    plt.ylabel(r'Accuracy (AUC)', fontsize = 16)
    
    # if task == "los_3": 
    #     plt.title(f"MIMIC3 Length of Stay GRU-D ExpM+NF AUC Distributions")
    # if task == "mort_icu": 
    #     plt.title(f"MIMIC3 Mortality GRU-D ExpM+NF AUC Distributions")
    
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    plt.savefig(Path(GRUD_PLOTS_PATH, task, 'expm_box.png'), bbox_inches='tight')
    print(task)
    plt.show()
    print()

# %% nf auc distribution plots 
colors = [sns.color_palette(palette='Purples', n_colors=6)[-3], 'purple']

for task in ['los_3', 'mort_icu']:
    folder = Path(GRUD_PLOTS_PATH, task, 'nf_auc_distributions') 
    folder.mkdir(exist_ok=True)
    for i, epsilon in enumerate([.001, .01, .1, 1]):
        df_e = df_expm[(df_expm.target == task ) & 
                       (df_expm.epsilon == epsilon)]
        df_e = df_e[df_e.seed.isin(df_e.seed.unique()[:2])]        
        sns.catplot(data = df_e, x = "epsilon", y = "aucs", kind = 'violin', hue = 'seed',
                    log_scale = (False ,False), bw_adjust=.5, cut=0, split=True, 
                    inner = None, linewidth = 0, legend = False, 
                    palette = colors)
        if i in [2,3]: plt.xlabel(r'Privacy ($\epsilon$)')
        else: plt.xlabel('')
        if i in [0, 2]: plt.ylabel(r'Accuracy (AUC)')
        else: plt.ylabel("")
        plt.savefig( Path(folder, f'{epsilon}.png'), bbox_inches= 'tight')

# # %% plot DPSGD bce vs l2 
# # epsilon 0.5 - 10:
# m, M= .5, 10 
# df_dpre = df_dpsgd[(df_dpsgd.target_epsilon >= m) & (df_dpsgd.target_epsilon <= M) ].sort_values('target_epsilon')
# add_baseline = False
# for task in ['los_3', 'mort_icu']:
#     df_d = df_dpre[df_dpre.target == task]
    
#     fig, ax = plt.subplots(layout = "constrained", figsize = (6.5, 5))
#     # plot dpsgd median  w/ bce loss 
#     sns.pointplot(data = df_d[df_d.loss == 'bce'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
#                 errorbar=None, estimator = 'median', color = 'green', scale= 1.5, 
#                 markers = 'o',  label = "BCE Loss")
#     # plot dpsgd plot w/ l2 loss
#     sns.pointplot(data = df_d[df_d.loss == 'l2'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
#                 errorbar=None, estimator = 'median', color = snot, markers = 'D', 
#                   label = "L2 Loss")#native_scale = True, 
    
#     ax.set_xlabel(r'Privacy ($\epsilon$)', fontsize = 16)
#     ax.set_ylabel( 'Accuracy (AUC)', fontsize = 16)

#     if add_baseline: 
#         ax.axhline(y=baseline[(task, "l2", "bn")], color="navy", linestyle="-", linewidth = 1, 
#                    label = f'baseline (no privacy), with batch norm, {baseline[task, "l2", "bn"]:.4}')
#         ax.axhline(y=baseline[(task, "l2", "nobn")], color="k", linestyle="--", linewidth = 1, 
#                    label = f'baseline (no privacy), no batch norm, {baseline[task, "l2", "nobn"]:.4}')
        
#     if task == 'los_3':
#         # plt.title(f"L2 vs. BCE Loss Median AUC Results\n MIMIC3 Length of Stay, GRU-D DPSGD $\delta=1E-5$")
#         if add_baseline: 
#             ym, yM = plt.ylim()
#             plt.ylim(.67, yM)

#     elif task == 'mort_icu': 
#         # plt.title(f"L2 vs. BCE Loss Median AUC Results\n MIMIC3 Mortality, GRU-D DPSGD $\delta=1E-5$")
#         if add_baseline:    
#             ym, yM = plt.ylim()
#             plt.ylim(.76, yM)

#     if add_baseline: 
#         leg = ax.legend()
#         leg_lines = leg.get_lines() # make legend baseline dotted
#         leg_lines[-1].set_linestyle(":")
#     # change the line width for the legend
#     handles, labels = ax.get_legend_handles_labels()
#     handles = [copy.copy(ha) for ha in handles ]
#     plt.legend(handles=handles, labels=labels, loc = 'lower right')
#     plt.yticks(fontsize=16)
#     plt.xticks(fontsize=16)

#     fig.savefig(Path(GRUD_PLOTS_PATH, task,  'bce_v_l2_epsilon_2e-1_to_10.png'), bbox_inches='tight')
#     plt.show()


# # %%
# m, M= .0001, 1 
# df_dpre = df_dpsgd[(df_dpsgd.target_epsilon >= m) & (df_dpsgd.target_epsilon <= M) & (df_dpsgd.target_epsilon != .5) ].sort_values('target_epsilon')
# add_baseline = False
# for task in ['los_3', 'mort_icu']:
#     df_d = df_dpre[df_dpre.target == task]

#     fig, ax = plt.subplots(layout = "constrained", figsize = (6.5, 5))
#     # plot dpsgd median  w/ bce loss 
#     sns.pointplot(data = df_d[df_d.loss == 'bce'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
#                 errorbar=None, estimator = 'median', color = 'green', scale= 1.5, 
#                 markers = 'o',  label = "BCE Loss")
#     # plot dpsgd plot w/ l2 loss
#     sns.pointplot(data = df_d[df_d.loss == 'l2'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
#                 errorbar=None, estimator = 'median', color = snot, markers = 'D', 
#                   label = "L2 Loss")#native_scale = True, 

#     ax.set_xlabel(r'Privacy ($\epsilon$), log scale', fontsize = 16)
#     ax.set_ylabel( 'Accuracy (AUC)', fontsize = 16)

#     if add_baseline: 
#         ax.axhline(y=baseline[(task, "l2", "bn")], color="navy", linestyle="-", linewidth = 1, 
#                    label = f'baseline (no privacy), with batch norm, {baseline[task, "l2", "bn"]:.4}')
#         ax.axhline(y=baseline[(task, "l2", "nobn")], color="k", linestyle="--", linewidth = 1, 
#                    label = f'baseline (no privacy), no batch norm, {baseline[task, "l2", "nobn"]:.4}')
        
#     if task == 'los_3':
#         # plt.title(f"L2 vs. BCE Loss Median AUC Results\n MIMIC3 Length of Stay, GRU-D DPSGD $\delta=1E-5$")
#         if add_baseline: 
#             ym, yM = plt.ylim()
#             plt.ylim(.67, yM)

#     elif task == 'mort_icu': 
#         # plt.title(f"L2 vs. BCE Loss Median AUC Results\n MIMIC3 Mortality, GRU-D DPSGD $\delta=1E-5$")
#         if add_baseline:    
#             ym, yM = plt.ylim()
#             plt.ylim(.76, yM)

#     if add_baseline: 
#         leg = ax.legend()
#         leg_lines = leg.get_lines() # make legend baseline dotted
#         leg_lines[-1].set_linestyle(":")
#     # change the line width for the legend
#     handles, labels = ax.get_legend_handles_labels()
#     handles = [copy.copy(ha) for ha in handles ]
#     plt.legend(handles=handles, labels=labels, loc = 'lower right')
#     plt.yticks(fontsize=16)
#     plt.xticks(fontsize=16)

#     fig.savefig(Path(GRUD_PLOTS_PATH, task,  'bce_v_l2_epsilon_1e-4_to_1.png'), bbox_inches='tight')
#     # plt.show()

# %%
