

import pandas as pd
import numpy as np
import sys, scipy
from pathlib import Path
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from config import *
from experiment_utils import *
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc, roc_curve
import functools
import statsmodels.api as sm
from scipy.stats import norm

# Look at me being proactive!
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dpsgd_color = '#00D91A' # dpsgd color
non_pr_color = '#284c70'
expm_color = 'purple'

# GPU setup 
n_procs = 2 # if using gpus, this will be per gpu
devices = {"proper_gpus": [0], "migs":[] }  # devices = {"proper_gpus": [], "migs":["mig1"] } 
gpu_names, device_ids, use_gpu = setup_devices(devices)
targets = ["mort_icu"]

def load_results(path, model_num):
    mia_df = pd.DataFrame.from_records(unjsonify(path))
    attack_scores = pd.DataFrame(mia_df.loc["attack_score"].to_list(), columns = range(0,len(mia_df["0"]["attack_score"])))[model_num].to_numpy()
    mems = pd.DataFrame(mia_df.loc["target_model_member"].to_list(), columns = range(0,len(mia_df["0"]["attack_score"])))[model_num].to_numpy()
    non_mems = pd.DataFrame(mia_df.loc["target_model_member"].to_list(), columns = range(0,len(mia_df["0"]["attack_score"])))[model_num].to_numpy()
    return attack_scores, mems

def load_results2(path):
    mia_df = pd.DataFrame.from_records(unjsonify(path))
    attack_scores = np.array([float(s) for s in mia_df.loc["attack_score"].explode().to_numpy()])
    mems = np.array([int(m) for m in mia_df.loc["target_model_member"].explode().to_numpy()])
    return attack_scores, mems

def sweep(score, x):
    # taken from https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    fpr, tpr, thresholds = roc_curve(x, -score)
    acc = np.max(1-(fpr+(1-tpr))/2)
    return fpr, tpr, auc(fpr, tpr), acc, thresholds

def do_plot(prediction, answers, legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    # taken from https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc, acc, _ = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    #lower = tpr[np.where(fpr<.001)[0][-1]] # in paper's code
    low = tpr[np.where(fpr<.1)[0][-1]] #what I think it should be

    print("------")
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f'%(legend, auc,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    plt.plot(fpr, tpr, label=legend+metric_text, **plot_kwargs)
    return (acc,auc)

methods = {"dpsgd": "DPSGD",
           "nonprivate": "Non-Private",
           "expm": "ExpM+NF"}
targets = {"mort_icu": "ICU Mortality Task",
           "los_3": "Length of Stay Task"}
losses = {"bce": "BCE",
        "l2": "l2"}

def count_correct(mems_preds, mems_true):
    correct = 0
    for mp in mems_preds:
        if mp in mems_true:
            correct += 1
    return correct

def most_accurate(fpr, tpr, thresholds, attack_scores, mems):
    #ideally, we'd use the highest TPR with FPR = 0, so we first try to find the last zero FPR, if that doesn't
    # exist then we'd like to use the first threshold that yields a non-zero tpr (since fpr is ordered this will 
    # correspond to a lowest false positive rate)
    i = np.nonzero(fpr)[0][0] - 1 # this is the last zero FPR since the FPR is ordered
    if i == -1: i = np.nonzero(tpr)[0][0]
    t = thresholds[i] 
    attack_preds = (-attack_scores >= t) # binary predictions
    nonmem_i = np.where(fpr == 1)[0][0] - 1
    nonmem_t = thresholds[nonmem_i]
    nonmem_attack_preds = (-attack_scores >= nonmem_t) # binary predictions

    mems_preds = np.nonzero(attack_preds)[0]
    mems_true = np.where(mems)[0]
    nonmems_preds = np.where(nonmem_attack_preds == 0)[0]
    nonmems_true = np.where(mems == 0)[0]
   
    # but we'd also like to at least return some members, so we won't test empty 
    # membership predictions 
    while len(mems_preds) == 0:
        i += 1 # increase the threshold by one step until we get at least one member prediction
        t = thresholds[i] 
        attack_preds = (-attack_scores >= t) # binary predictions
        # these are the data points that our attack said were members
        mems_preds = np.nonzero(attack_preds)[0]

    return i, mems_preds, mems_true, count_correct(mems_preds, mems_true), nonmems_preds, nonmems_true, count_correct(nonmems_preds, nonmems_true)



def at_most_one_wrong(fpr, tpr, thresholds, attack_scores, mems):
    # here we'll see how many in total can be reidentified but with the highest accuracy
    # so what we'll do is keep changing the threshold hold until we get more than 2 incorrect 
    # predictions

    # we'll start at the last non-zero FPR (or the lowest if there is no non-zero FPR) and incremement until we 
    # get 2 incorrect predictions
    i = np.nonzero(fpr)[0][0] - 1 # this is the last zero FPR since the FPR is ordered
    if i == -1: i = np.nonzero(tpr)[0][0]
    t = thresholds[i] 
    attack_preds = (-attack_scores >= t) # binary predictions

    mems_preds = np.nonzero(attack_preds)[0]
    mems_true = np.where(mems)[0]
    nonmems_preds = np.where(attack_preds == 0)[0]
    nonmems_true = np.where(mems == 0)[0]
    num_mem_incorrect = len(mems_preds)-count_correct(mems_preds, mems_true)
    num_nonmem_incorrect = len(nonmems_preds)-count_correct(nonmems_preds, nonmems_true)
    
    while num_incorrect < 2:
        i += 1 # increase the threshold by one step until we get at least one member prediction
        t = thresholds[i] 
        attack_preds = (-attack_scores >= t) # binary predictions

        # these are the data points that our attack said were members
        mems_preds = np.nonzero(attack_preds)[0]
        nonmems_preds = np.where(attack_preds == 0)[0]
        num_mems_incorrect = len(mems_preds)-count_correct(mems_preds, mems_true)
        num_nonmem_incorrect = len(nonmems_preds)-count_correct(nonmems_preds, nonmems_true)

    # this i will actually give 2 incorrect so we will go back one:
    i -= 1
    t = thresholds[i] 
    attack_preds = (-attack_scores >= t) # binary predictions
    mems_preds = np.nonzero(attack_preds)[0]
    nonmems_preds = np.where(attack_preds == 0)[0]
    # unless that gives us no members then we'll go back to i
    if len(mems_preds) == 0:
        t = thresholds[i+1] 
        attack_preds = (-attack_scores >= t) # binary predictions
        mems_preds = np.nonzero(attack_preds)[0]
        nonmems_preds = np.where(attack_preds == 0)[0]

    return i, mems_preds, mems_true, count_correct(mems_preds, mems_true), nonmems_preds, nonmems_true, count_correct(nonmems_preds, nonmems_true)

def most_reidentified(fpr, tpr, thresholds, attack_scores, mems):
    # here we'll see how many in total can be reidentified but with the highest accuracy
    # so what we'll do is keep changing the threshold hold until we get more than 2 incorrect 
    # predictions
    mems_true = np.where(mems)[0]
    nonmems_true = np.where(mems == 0)[0]
    reid_rate = []
    inds = np.nonzero(tpr)[0] # so we don't divide by 0
    to_use_tpr = tpr[inds]
    to_use_fpr = fpr[inds]
    to_use_thresholds = thresholds[inds]
    
    total_correct = len(mems_true)*to_use_tpr
    total_pred = total_correct + len(mems_true)*to_use_fpr
    acc = total_correct/total_pred
    reid_rate = total_correct/len(mems_true) 
    
    # get every place where the accuracy is higher than 
    tol_acc = np.where(acc >= .8)[0]
    if len(tol_acc) == 0:
        tol_acc = [np.argmax(acc)]

    to_use_i = np.argmax(reid_rate[tol_acc])
    # find the real threshold index
    to_use_t = to_use_thresholds[tol_acc][to_use_i] 
    i = np.where(thresholds == to_use_t)[0][0]
    t = thresholds[i]
    assert t == to_use_t

    attack_preds = (-attack_scores >= to_use_t) # binary predictions

    # these are the data points that our attack said were members
    mems_preds = np.nonzero(attack_preds)[0]
    nonmems_preds = np.where(attack_preds == 0)[0]

    # we'd also like to at least return some members, so we won't test empty 
    # membership predictions 
    while len(mems_preds) == 0:
        i += 1 # increase the threshold by one step until we get at least one member prediction
        t = thresholds[i] 
        attack_preds = (-attack_scores >= t) # binary predictions
        # these are the data points that our attack said were members
        mems_preds = np.nonzero(attack_preds)[0]
        nonmems_preds = np.where(attack_preds == 0)[0]

    return i, mems_preds, mems_true, count_correct(mems_preds, mems_true), nonmems_preds, nonmems_true, count_correct(nonmems_preds, nonmems_true)

    

def analyze_attack(run, method, target, loss, file_path, model_num, epsilon = "", print_attack_name = False,
                   analysis_fn = most_accurate):

    global methods, targets, losses

    if epsilon == "":
        label = f"{methods[method]} ({losses[loss]} Loss)"
    else:
        label = f"{methods[method]} ({losses[loss]} Loss, eps = {epsilon})"
    mia_path = Path(LR_MIA_RESULTS_FOLDER, f"run_{run}", f"{method}", f"{target}_loss_{loss}_results", file_path)

    attack_scores, mems = load_results(mia_path, model_num)
    fpr, tpr, auc, acc, thresholds = sweep(np.array(attack_scores), np.array(mems, dtype=bool))

    i, mems_preds, mems_true, mems_correct, nonmems_preds, nonmems_true, nonmems_correct = analysis_fn(fpr, tpr, thresholds, attack_scores, mems)
   
    def get_perc(n, d): return round(n/d*100, 4)
    if print_attack_name:
        print("------")
        print(f"Attack {label}")
    print(f"   total members: {len(mems_true)}, total members predicted: {len(mems_preds)}, total correct: {mems_correct}\n    tpr: {tpr[i]}, fpr: {fpr[i]}, PPV: {get_perc(mems_correct, len(mems_preds))}%, re-identified: {get_perc(mems_correct, len(mems_true))}%")
    return mems_preds, mems_true

def fig_max_ppv_hist(fig, ax, run, method, target, loss, file_path, epsilon = "", save = False,
                model_nums = range(0,10), **plot_kwargs):
    # taken from https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py

    global methods, targets, losses

    mia_path = Path(LR_MIA_RESULTS_FOLDER, f"run_{run}", f"{method}", f"{target}_loss_{loss}_results", file_path)
       
    max_ppvs = []
    for model_num in model_nums:
        if epsilon == "":
            label = f"{methods[method]} ({losses[loss]} Loss)\n"
        else:
            label = f"{methods[method]} ({losses[loss]} Loss, eps = {epsilon})\n"
     
        attack_scores, mems = load_results(mia_path, model_num)
        fpr, tpr, auc, _, thresholds = sweep(np.array(attack_scores), np.array(mems, dtype=bool))

        mems_true = np.where(mems)[0]
        inds = np.nonzero(tpr)[0] # so we don't divide by 0
        to_use_tpr = tpr[inds]
        to_use_fpr = fpr[inds]
        to_use_thresholds = thresholds[inds]
        total_correct = len(mems_true)*to_use_tpr
        total_pred = total_correct + len(mems_true)*to_use_fpr
        ppv = total_correct/total_pred
     
        max_ppvs.append(np.max(ppv))

    fig.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.hist(max_ppvs, label = label)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
    if save:
        fig.savefig(Path(LR_MIA_PLOTS_PATH, f"{methods[method]}_{target}_epsilon_{epsilon}_attack_auc_histogram.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')



def fig_fpr_tpr(fig, ax, run, method, target, loss, file_path, epsilon = "", save = False,
                model_nums = range(0,10), analysis_fn = most_accurate, **plot_kwargs):
    # taken from https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py

    global methods, targets, losses

    mia_path = Path(LR_MIA_RESULTS_FOLDER, f"run_{run}", f"{method}", f"{target}_loss_{loss}_results", file_path)
       
    for model_num in model_nums:
        if epsilon == "":
            label = f"{methods[method]} ({losses[loss]} Loss)\n"
        else:
            label = f"{methods[method]} ({losses[loss]} Loss, eps = {epsilon})\n"
     
        attack_scores, mems = load_results(mia_path, model_num)
        do_plot(attack_scores, mems, 
                legend = label,
                metric='auc',
                **plot_kwargs
        )
        analyze_attack(run, method, target, loss, file_path, model_num, epsilon, analysis_fn = analysis_fn)
    ax.semilogx()
    ax.semilogy()
    ax.plot([0, 1], [0, 1], ls='--', color='gray')
    fig.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, fontsize = 8, loc='center left', bbox_to_anchor=(1, 0.5))
    if analysis_fn == most_accurate:
        analysis_label = "most_accurate"
    elif analysis_fn == most_reidentified:
        analysis_label = "most_reidentified"
    elif analysis_fn == at_most_one_wrong:
        analysis_label = "most_reidentified"
    if save:
        fig.savefig(Path(LR_MIA_PLOTS_PATH, f"{analysis_label}_{target}_epsilon_{epsilon}_fpr_tpr.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')


def qqplot(Y, ax, **plt_kwargs):   
    Y = sorted(Y)
    N = len(Y)
    x = [norm.ppf(i/(N+1)) for i in range(1, N+1)] # = { CDF^{-1}(i/(N+1) } 
    ## to compare to y = x: 
    line_values = np.arange(x[0], x[-1], .1)
    ax.plot(line_values, line_values, color = 'black', linestyle = '-.')

    # plot our percentiles against each other: 
    ax.scatter(x, Y,  marker = "o", **plt_kwargs)
    plt.ylabel("Sample's Empirical Quantiles" )
    plt.xlabel("Standard Normal's Quantiles")

def fig_shadow_confs(run, target, loss, data_inds = [], save = True):
    shadow_model_path = Path(LR_MIA_RESULTS_FOLDER, f"run_{run}", "shadow_models",
                             f"{target}_loss_{loss}_results", "mia_shadow_confs.json")
    shadow_df = pd.DataFrame.from_records(unjsonify(shadow_model_path))

    for ind in data_inds:
        confs_in = np.array(shadow_df[f"{ind}"]["confs_in"]).squeeze()
        confs_out = np.array(shadow_df[f"{ind}"]["confs_out"]).squeeze()
        
        fig = plt.figure(figsize=(9,6))
        gs = GridSpec(2, 2, figure=fig, width_ratios = [2, 1])
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])

#        fig.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
        ax1.hist(confs_in, label = "confs when member", alpha = .5, color = 'blue')
        ax1.hist(confs_out, label = "confs when non-member", alpha = .5, color = 'green')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, fontsize = 8, loc='upper left', bbox_to_anchor=(.95, -0.1))
    

        # qq plot
        confs_in_norm = (np.array(shadow_df[f"{ind}"]["confs_in"]) - shadow_df[f"{ind}"]["confs_in_mean"])/shadow_df[f"{ind}"]["confs_in_std"]
        confs_out_norm = (np.array(shadow_df[f"{ind}"]["confs_out"]) - shadow_df[f"{ind}"]["confs_out_mean"])/shadow_df[f"{ind}"]["confs_out_std"]
#        confs_in_norm = np.random.normal(0,1, 1000)
#        confs_out_norm = np.random.normal(0,1, 1000)
#        sm.qqplot(confs_in_norm, line='45', ax = ax2, 
#                  markerfacecolor = 'blue', markeredgecolor = "blue",
#                  alpha = .5)
#        sm.qqplot(confs_out_norm, line='45', ax = ax3,
#                  markerfacecolor = 'green', markeredgecolor = "green",
#                  alpha = .5)
        qqplot(confs_in_norm, ax = ax2, color = "blue")
        qqplot(confs_out_norm, ax = ax3, color = "green")
        if save:
            fig.savefig(Path(LR_MIA_PLOTS_PATH, "shadow_models", f"datapoint_{ind}_{run}_{target}_loss_{loss}.png"), bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)



if __name__ == '__main__':


#    print("------ Most Accurate Attacker ------")
    # most accurate
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", 
#                model_nums = [4], color = non_pr_color)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "dpsgd", "mort_icu", "bce", "mia_dpsgd_epsilon_0.0001.json", ".0001", 
#                model_nums = [2], color = dpsgd_color)
#    fig_fpr_tpr(fig, ax, "stable_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_0.0001.json", ".0001", True,
#                model_nums = [5], color = expm_color)
#    plt.close(fig)
#
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", 
#                model_nums = [4], color = non_pr_color)
#    fig_fpr_tpr(fig, ax, "stable_1000", "dpsgd", "mort_icu", "bce", "mia_dpsgd_epsilon_0.1.json", ".1", 
#                model_nums = [8], color = dpsgd_color)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_0.1.json", ".1", True,
#                model_nums = [7], color = expm_color)
#    plt.close(fig)
#
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", 
#                model_nums = [4], color = non_pr_color)
#    fig_fpr_tpr(fig, ax, "stable_1000", "dpsgd", "mort_icu", "bce", "mia_dpsgd_epsilon_7.json", "7", 
#                model_nums = [5], color = dpsgd_color)
#    fig_fpr_tpr(fig, ax, "stable_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_7.json", "7", True,
#                model_nums = [7], color = expm_color)
#    plt.close(fig)

    print("\n\n------ Most Reidentified (with highest accuracy) ------")
    # less accurate attacker but higher reidentification rate
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    fig_fpr_tpr(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", 
                model_nums = [0], color = non_pr_color, analysis_fn = most_reidentified)
    fig_fpr_tpr(fig, ax, "hinge_1000", "dpsgd", "mort_icu", "bce", "mia_dpsgd_epsilon_0.0001.json", ".0001", 
                model_nums = [4], color = dpsgd_color, analysis_fn = most_reidentified)
    fig_fpr_tpr(fig, ax, "stable_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_0.0001.json", ".0001", True,
                model_nums = [5], color = expm_color, analysis_fn = most_reidentified)
    plt.close(fig)

#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", 
#                model_nums = [0], color = non_pr_color, analysis_fn = most_reidentified)
#    fig_fpr_tpr(fig, ax, "stable_1000", "dpsgd", "mort_icu", "bce", "mia_dpsgd_epsilon_0.1.json", ".1", 
#                model_nums = [8], color = dpsgd_color, analysis_fn = most_reidentified)
#    fig_fpr_tpr(fig, ax, "stable_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_0.1.json", ".1", True,
#                model_nums = [3], color = expm_color, analysis_fn = most_reidentified)
#    plt.close(fig)
#
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", 
#                model_nums = [0], color = non_pr_color, analysis_fn = most_reidentified)
#    fig_fpr_tpr(fig, ax, "stable_1000", "dpsgd", "mort_icu", "bce", "mia_dpsgd_epsilon_7.json", "7", 
#                model_nums = [5], color = dpsgd_color, analysis_fn = most_reidentified)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_7.json", "7", True,
#                model_nums = [5], color = expm_color, analysis_fn = most_reidentified)
#    plt.close(fig)
#

#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_max_acc_hist(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", 
#                     "mia_nonprivate_baseline.json", save = True, color = non_pr_color)
#    plt.close(fig)
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_max_acc_hist(fig, ax, "stable_1000", "dpsgd", "mort_icu", "bce", 
#                     "mia_dpsgd_epsilon_7.json", "7", True, color = dpsgd_color)
#    plt.close(fig)
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_max_acc_hist(fig, ax, "hinge_1000", "expm", "mort_icu", "l2", 
#                    "mia_expm_epsilon_7.json", "7", True, color = expm_color)
#    plt.close(fig)
#

    # confidence plots for individual data points
#    fig_shadow_confs("hinge_1000", "mort_icu", "bce", data_inds = [0, 10, 20], save = True)
#    fig_shadow_confs("stable_1000", "mort_icu", "bce", data_inds = [0, 10, 20], save = True)








    # helpful for looking at all the 10 model results
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "stable_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", "1", True,
#                model_nums = range(0,5), analysis_fn = most_reidentified)
#    plt.close(fig)
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "stable_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", "2", True,
#                model_nums = range(5, 10), analysis_fn = most_reidentified)
#    plt.close(fig)
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", "3", True,
#                model_nums = range(0,5), analysis_fn = most_reidentified)
#    plt.close(fig)
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "nonprivate", "mort_icu", "bce", "mia_nonprivate_baseline.json", "4", True,
#                model_nums = range(5, 10), analysis_fn = most_reidentified)
#    plt.close(fig)

#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "stable_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_0.0001.json", "1", True,
#                model_nums = range(0,5), analysis_fn = most_reidentified)
#    plt.close(fig)
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "stable_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_0.0001.json", "2", True,
#                model_nums = range(5, 10), analysis_fn = most_reidentified)
#    plt.close(fig)
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_0.0001.json", "3", True,
#                model_nums = range(0,5), analysis_fn = most_reidentified)
#    plt.close(fig)
#    fig = plt.figure(figsize=(4,3))
#    ax = fig.add_subplot(111)
#    fig_fpr_tpr(fig, ax, "hinge_1000", "expm", "mort_icu", "l2", "mia_expm_epsilon_0.0001.json", "4", True,
#                model_nums = range(5, 10), analysis_fn = most_reidentified)
#    plt.close(fig)
#
