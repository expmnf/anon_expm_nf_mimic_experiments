### plots lr final results. 

# %%
import sys, copy 
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(Path(".", "code").absolute().as_posix())
sys.path.append(Path(".", "code", "mimic_experiment").absolute().as_posix())
from config import * #experiment configuration variables & results path
from experiment_utils import * # this loads get_auc()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #supress future warnings
warnings.simplefilter(action='ignore', category=UserWarning) #supress user warnings
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

snot = '#00D91A' # dpsgd color


# %%
df_nonpriv = pd.DataFrame.from_records(
    map(unjsonify, Path(LR_BASELINE_RESULTS_FOLDER, 'final0' ).glob('**/*.json')))[['auc', 'seed', 'loss', 'target']]
df_nonpriv['loss'] = df_nonpriv.loss.fillna('l2')
baseline = df_nonpriv.groupby(['target', 'loss']).median()['auc'] # Series w/ median AUC for each task
# baseline.loc['loc_3'] ## gives the median auc for los

# indicator for if we want to plot the baseline or not 
add_baseline = True # change to true when we have the baseline done. 

df_dpsgd = pd.DataFrame.from_records(
    map(unjsonify,Path(LR_DPSGD_RESULTS_FOLDER , 'final0')
        .glob('**/*.json')))\
            [['target', 'target_epsilon', 'loss', 'auc']]\
                .rename(columns={'auc':'aucs'})
# df_dpsgd['loss_fn'] = df_dpsgd['loss_fn'].fillna('l2')

df_expm = pd.DataFrame.from_records(
    map(unjsonify, Path(LR_EXPM_RESULTS_FOLDER , 'final0').glob('**/*.json')))[['target', 'epsilon', 'aucs', 'seed']].explode('aucs')

# print(sorted(df_dpsgd.target_epsilon.unique()))
## for table, print the baselines: 
print(baseline)

# %% 
# make folder for plots: 
print("making folders for plots")
for task in ['los_3', 'mort_icu']:
    Path(LR_PLOTS_PATH, task).mkdir(exist_ok=True, parents=True)
    
# %% 
# epsilon 0.5 - 10:
m, M= .5, 10 
df_epre = df_expm[(df_expm.epsilon >= m) & (df_expm.epsilon <= M) ].sort_values('epsilon')
df_dpre = df_dpsgd[(df_dpsgd.target_epsilon >= m) & (df_dpsgd.target_epsilon <= M) ].sort_values('target_epsilon')

for task in ['los_3', 'mort_icu']:
    print(f"making epsilon in {[m,M]} plots for {task}")
    df_e = df_epre[df_epre.target == task]
    df_d = df_dpre[df_dpre.target == task]
    # makes expm plot w/ median markers

    fig, ax = plt.subplots(layout = "constrained", figsize=(6.5, 5))
    # makes dpsgd plot w/ diamonds at median
    sns.pointplot(data = df_d[df_d.loss == 'l2'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = snot, 
                markers = 'D',  label = r'DPSGD, $\ell(2)$ loss, $\delta=1E-5$')
    sns.pointplot(data = df_d[df_d.loss == 'bce'], ax = ax, x = "target_epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = 'green', scale = 1.25,
                markers = 'o',  label = r'DPSGD, BCE loss, $\delta=1E-5$')
    sns.pointplot(data = df_e, ax = ax, x = "epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = 'purple', 
                markers = 'D',  label = "ExpM+NF")#native_scale = True, 


    ax.set_xlabel(r'Privacy ($\epsilon$)', fontsize = 16)
    ax.set_ylabel( 'Accuracy (AUC)', fontsize = 16)

    if add_baseline: 
        ax.axhline(y=baseline[task].max(), color="k", linestyle="--", linewidth = 1, 
                   label = f'baseline (no privacy), {baseline[task].max():.4}')
    if task == 'los_3':
        pass
    elif task == 'mort_icu': 
        ym, yM = plt.ylim()
        plt.ylim(.59, yM)
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
    spath = Path(LR_PLOTS_PATH, task,  'epsilon_2e-1_to_10.png')
    fig.savefig(spath, bbox_inches='tight')
    print(f"saved to {spath}")
    #plt.show()
    print()



# %% 
## Now epsilon .01, 1, log plot. 
m, M= .0001, 1 
df_epre = df_expm[(df_expm.epsilon >= m) & (df_expm.epsilon <= M) & (df_expm.epsilon != .5)].sort_values('epsilon')
df_dpre = df_dpsgd[(df_dpsgd.target_epsilon >= m) & (df_dpsgd.target_epsilon <= M) & (df_dpsgd.target_epsilon != .5) ].sort_values('target_epsilon')

for task in ['los_3', 'mort_icu']:
    print(f"making epsilon in {[m,M]} plots for {task}")
    df_e = df_epre[df_epre.target == task]
    df_d = df_dpre[df_dpre.target == task]

    # makes expm plot w/ median markers
    fig, ax = plt.subplots(layout = "constrained", figsize = (6.5, 5))
    # makes dpsgd plots
    sns.pointplot(data = df_d[df_d.loss == "l2"], ax = ax, x = "target_epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = snot, native_scale = True, 
                markers = 'D', label = r'DPSGD, $\ell(2)$ loss $\delta=1E-5$')
    
    sns.pointplot( data = df_d[df_d.loss == "bce"], ax = ax, x = "target_epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', native_scale = True,  
                color = 'green', scale = 1.25,
                markers = 'o',  label = r'DPSGD, BCE loss, $\delta=1E-5$')
    sns.pointplot(data = df_e, ax = ax, x = "epsilon", y = "aucs", join = False,
                errorbar=None, estimator = 'median', color = 'purple', 
                markers = 'D', label = "ExpM+NF", native_scale = True)

    if add_baseline: ax.axhline(y=baseline[task].max(), color="k", 
                    linestyle="--", linewidth = 1, 
                    label = f'baseline (no privacy), {baseline[task].max():.4}')
    ax.set_xscale('log')
    ax.set_xlabel(r'Privacy ($\epsilon$), log scale', fontsize = 16)
    ax.set_ylabel( 'Accuracy (AUC)', fontsize = 16)
    # if task == 'los_3':
    #     plt.title(f"MIMIC3 Length of Stay LR ExpM+NF & DPSGD Median AUC Results")
    # elif task == 'mort_icu': 
    #     plt.title((f"MIMIC3 Mortality LR ExpM+NF & DPSGD Median AUC Results"))

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
    fig.savefig(Path(LR_PLOTS_PATH, task, 'epsilon_1e-4_to_1.png'), bbox_inches='tight')
    # plt.show()()
    print()

# %% box plots 
#m, M= .0001, 10 
m, M= 1e-4, 10 
df_epre = df_expm[df_expm.epsilon.isin([1e-6, 1e-5, .0001, .001, .01, .1, 1, 10])].sort_values('epsilon')
print("starting box plots...")
for task in ['los_3', 'mort_icu']:
    df_e = df_epre[df_epre.target == task]
    df_e = df_e.astype({"aucs": float})
    df_eg = df_e.groupby(['epsilon', 'seed']).mean(numeric_only=True).reset_index()
    ## makes expm box plots
    sns.catplot( data = df_e, x = "epsilon", y = "aucs", kind = 'box', 
                    showmeans=False, showfliers=False, width=0.37, height=5, aspect=6.5/5,
                    medianprops={"color": "purple", "linewidth": 2},
                    boxprops={"facecolor": "white", "edgecolor": "purple",
                            "linewidth": 0.5, 'alpha' : .5},
                    whiskerprops={"color": "purple", "linewidth": 2, 'alpha': .5},
                    capprops={"color": "purple", "linewidth": 1.5, 'alpha':.5}, 
                     legend='auto', native_scale=True, log_scale = (True ,False))
    ## adds the 10 models' means to show that some runs did poorly and pulled down the distribution 
    sns.swarmplot( data = df_eg,  x = "epsilon", y = "aucs", 
                 alpha = .5, legend = True, color ='purple', size = 3,
                 native_scale=True, log_scale = (True ,False) )
    plt.xlabel(r'Privacy ($\epsilon$), log scale', fontsize = 16)
    plt.ylabel(r'Accuracy (AUC)', fontsize = 16)
    
    # if task == "los_3": 
    #     plt.title(f"MIMIC3 Length of Stay LR ExpM+NF AUC Distributions", fontsize = 16)
    # if task == "mort_icu": 
    #     plt.title(f"MIMIC3 Mortality LR ExpM+NF AUC Distributions", fontsize = 16)
    
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    print(task)
    plt.savefig(Path(LR_PLOTS_PATH, task, 'expm_box.png'), bbox_inches='tight')
    # plt.show()()
    print()


# %% NF AUC densities
colors = [sns.color_palette(palette='Purples', n_colors=6)[-3], 'purple']

for task in ['los_3', 'mort_icu']:
    print("starting NF AUC Density Plots ")
    folder = Path(LR_PLOTS_PATH, task, 'nf_auc_distributions') 
    folder.mkdir(exist_ok=True)
    print(f"plotting NF AUC desnities for task {task}")
    for i, epsilon in enumerate([.001, .01, .1, 1]):
        print(f"\t epsilon = {epsilon}")
        df_e = df_expm[(df_expm.target == task ) & 
                       (df_expm.epsilon == epsilon)]
        df_e = df_e[df_e.seed.isin(df_e.seed.unique()[:2])]
        df_e = df_e.astype({"aucs": float})        
        sns.catplot(data = df_e, x = "epsilon", y = "aucs", kind = 'violin', hue = 'seed',
                     bw_adjust=.5, cut=0, split=True, 
                    inner = None, linewidth = 0, legend = False, 
                    palette = colors,  log_scale = (False ,False))
        if i in [2,3]: plt.xlabel(r'Privacy ($\epsilon$)')
        else: plt.xlabel('')
        if i in [0, 2]: plt.ylabel(r'Accuracy (AUC)')
        else: plt.ylabel("")
        plt.savefig( Path(folder, f'{epsilon}.png'), bbox_inches= 'tight')
        # plt.show()()
        print()
# %%
print(baseline)

print(f"done! plots are in {LR_PLOTS_PATH.as_posix()} ")


