import pandas as pd, numpy as np, sys, scipy, itertools, math
from pathlib import Path
sys.path.append(Path('code').as_posix())
sys.path.append(Path('code', 'mimic_experiment').as_posix())
from config import *
from experiment_utils import *
import scipy.stats
import matplotlib.pyplot as plt, seaborn as sns, matplotlib.colors as colors, matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
from sklearn.metrics import auc, roc_curve
import functools
import statsmodels.api as sm
from scipy.stats import norm

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rcParams['text.usetex'] = True # TeX rendering

dpsgd_color = '#00D91A' # dpsgd color
non_pr_color = '#284c70'
expm_color = '#8c27b0'#'purple'

light_colors = {expm_color: '#bc95c9',
                dpsgd_color: '#addbb3',
                non_pr_color: '#99b3cc'}

methods = {"dpsgd": "DPSGD",
           "nonprivate": "Non-Private",
           "expm": "ExpM+NF"}
targets = {"mort_icu": "ICU Mortality Task",
           "los_3": "Length of Stay Task"}
losses = {"bce": "BCE",
        "l2": "l2"}
stats = {'tpr': "True Positive Rate",
         'thresholds': "Thresholds",
         'ppv': "Positive Predictive Value (PPV)",
         'npv': "Negative Predictive Value (NPV)",
         'mem_correct_pred': "Members Correctly Predicted",
         'nonmem_correct_pred': "Non-Members Correctly Predicted",
         'auc': "Attack AUC",
         'tpr_at_tenth_fpr': "TPR at .1\%FPR",
         'tpr_at_hundredth_fpr': "TPR at .001\%FPR",
        }
stats_acr = {'tpr': "TPR",
         'ppv': "PPV",
         'npv': "NPV",
         'auc': "AUC",
        }


########## Load and process data

def attack_stats(attack_scores, is_mems):    
    is_mems = is_mems.to_numpy(dtype = bool)
    attack_scores = attack_scores.to_numpy()
    nmems = len(np.where(is_mems)[0])
    nnonmems = len(is_mems) - nmems
    fpr, tpr, thresholds = roc_curve(is_mems, -attack_scores)
   
    hund = tpr[np.where(fpr<.00001)[0][-1]] if len(np.where(fpr<.00001)[0]) > 1 else 0
    tenth = tpr[np.where(fpr<.001)[0][-1]] if len(np.where(fpr<.001)[0]) > 1 else 0
    return pd.DataFrame({'fpr': fpr, 
            'tpr': tpr, 
            'thresholds': thresholds, 
            'ppv': tpr/(tpr+fpr),
            'npv': (1-fpr)/((1-fpr)+(1-tpr)),
            'mem_correct_pred': nmems*tpr,
            'nonmem_correct_pred': nnonmems * (1-fpr),
            'auc': auc(fpr, tpr),
            'tpr_at_tenth_fpr': tenth,
            'tpr_at_hundredth_fpr': hund})

def attack_stats_df(mia_df):
    df = mia_df.groupby('model_id').apply(lambda cols: attack_stats(cols.attack_score, cols.is_mem), include_groups = False)
    df = df.reset_index().drop(columns = 'level_1')
    df['model_id'] = list(map(int, df.model_id))
    return df

def audit(audit_scores, k_pos, k_neg):
    sorted_inds = np.argsort(audit_scores)
    T = np.zeros(len(audit_scores))
    T[sorted_inds[0:k_pos]] = 1
    T[sorted_inds[::-1][0:k_neg]] = -1

    return pd.Series(T)

def audit_df(mia_df, k_pos = None, k_neg = None):
    has_eps = "epsilon" in mia_df.columns
    def k_pos_fn(cols): 
        if k_pos == None: return cols.is_mem.sum()
        return k_pos
    def k_neg_fn(cols): 
        if k_neg == None: return (~cols.is_mem).sum()
        return k_neg
    # count how many correct guesses 
    def _count_pos_correct(guesses, is_mems):
        is_mems = is_mems.to_numpy()
        return len(np.intersect1d(np.where(is_mems == True), np.where(guesses == 1)))
    def _count_neg_correct(guesses, is_mems):
        is_mems = is_mems.to_numpy()
        return len(np.intersect1d(np.where(is_mems == False), np.where(guesses == -1))) 

    if has_eps:
        df = mia_df.groupby(["model_id", "epsilon"]).apply(lambda cols: {"audit_guesses": audit(cols.audit_score, k_pos_fn(cols), k_neg_fn(cols)).to_numpy(), "is_mem": cols.is_mem.to_numpy(), "data_id": cols.data_id.to_numpy()}).reset_index(name = "dicts")
        df = pd.concat([df.drop(columns = "dicts"), df['dicts'].apply(pd.Series).explode(["audit_guesses", "is_mem", "data_id"])], axis = 1)
    else:
        df = mia_df.groupby("model_id").apply(lambda cols: audit(cols.audit_score, k_pos_fn(cols), k_neg_fn(cols))).T
        df["data_id"] = mia_df.data_id
        df = pd.melt(df, id_vars = "data_id", value_name = "audit_guesses", var_name = "model_id").merge(mia_df).drop(columns = ["attack_score", "audit_score"])
        df["epsilon"] = "Inf"

    df = df.groupby(["model_id", "epsilon"]).apply(lambda cols: {"correct_pos_guesses": _count_pos_correct(cols.audit_guesses, cols.is_mem), "correct_neg_guesses": _count_neg_correct(cols.audit_guesses, cols.is_mem),"total_guesses": len(cols.audit_guesses.loc[lambda x: x != 0]), "total_examples": len(cols.is_mem)}).reset_index(name = "dicts")
    df = pd.concat([df.drop(columns = "dicts"), df['dicts'].apply(pd.Series)], axis = 1)
    df["correct_guesses"] = df.groupby(["model_id", "epsilon"]).apply(lambda cols: sum(cols.correct_pos_guesses, cols.correct_neg_guesses)).reset_index(name = "correct_guesses").correct_guesses

    return df
    
def p_value_DP_audit(m, r, v, eps, delta): 
    assert 0 <= v <= r <= m 
    assert eps >= 0 
    assert 0 <= delta <= 1 
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response 
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v] 
    alpha = 0
    s = 0 # = P[v > Binomial(r, q) >= v - i] 
    for i in range(1, v + 1): 
        s = s + scipy.stats.binom.pmf(v - i, r, q) 
        if s > i * alpha: 
            alpha = s / i 
    p = beta + alpha * delta * 2 * m 
    return min(p, 1)

# m = number of examples, each included independently with probability 0.5 
# r = number of guesses (i.e. excluding abstentions) 
# v = number of correct guesses by auditor 
# p = 1-confidence e.g. p=0.05 corresponds to 95%
def get_eps_audit(m, r, v, delta, p): 
    assert 0 <= v <= r <= m 
    assert 0 <= delta <= 1 
    assert 0 < p < 1 
    eps_min = 0 # maintain p_value_DP(eps_min) < p 
    eps_max = 1 # maintain p_value_DP(eps_max) >= p
     
    while p_value_DP_audit(m, r, v, eps_max, delta) < p: 
        eps_max = eps_max + 1 
    for _ in range(30): # binary search 
        eps = (eps_min + eps_max) / 2 
        if p_value_DP_audit(m, r, v, eps, delta) < p: 
            eps_min = eps 
        else: 
            eps_max = eps 
    return eps_min

def get_lower_eps(audit_df, delta = 0, p = 1-.95):
    def _get_eps_low(correct_guesses, total_guesses, total_examples):
        # since the eps audit is monotonically increasing, if the number of guesses
        # and total examples are the same across all models then we only need to 
        # run it once
        if len(total_guesses.unique()) == 1 and len(total_examples.unique()) == 1:
            m = total_examples.unique()[0]
            r = total_guesses.unique()[0]    
            v = round(np.quantile(correct_guesses.to_numpy(), p))
            return get_eps_audit(m, r, v, delta, p)
        else:
            raise Exception("Models did not have the same number of guesses/examples")
    df = audit_df.groupby("epsilon").apply(lambda cols: _get_eps_low(cols.correct_guesses, cols.total_guesses, cols.total_examples)).reset_index(name = "audit_epsilon_low")
    df["audit_delta"] = delta
    return df

def get_result_df(run, method, N_target_models, target, loss, file_path, mia_results_path, add_to_path):

    if N_target_models == "":
        mia_path = Path(mia_results_path, f"run_{run}", f"{method}", f"{target}_loss{loss}_results", file_path)
    else:
        mia_path = Path(mia_results_path, f"run_{run}", f"{method}_{N_target_models}", f"{target}_loss{loss}_{add_to_path}results", file_path)

    mia_df = pd.DataFrame.from_records(unjsonify(mia_path)).T
    mia_df = mia_df.drop('conf_obs', axis = 1) # not needed
    att = mia_df.attack_score.apply(pd.Series)
    att["data_id"] = att.index
    tar = mia_df.target_model_member.apply(pd.Series)
    tar["data_id"] = tar.index
    aud = mia_df.audit_score.apply(pd.Series)
    aud["data_id"] = aud.index
    return pd.melt(att, id_vars = "data_id", value_name = "attack_score", var_name = "model_id").merge(pd.melt(tar, id_vars = "data_id", value_name = "is_mem", var_name = "model_id")).merge(pd.melt(aud, id_vars = "data_id", value_name = "audit_score", var_name = "model_id"))

def get_target_model_df(run, method, N_target_models, target, loss, mia_results_path, add_to_path, epsilon_str = ""):

    if N_target_models == "":
        mia_path = Path(mia_results_path, f"run_{run}", f"{method}", "hyperparams", f"{target}_loss{loss}_results")
    else:
        if method == "expm":
            mia_path = Path(mia_results_path, f"run_{run}", f"{method}_{N_target_models}", "hyperparams", f"{target}_{epsilon_str}results")
        else:
            mia_path = Path(mia_results_path, f"run_{run}", f"{method}_{N_target_models}", "hyperparams", f"{target}_{epsilon_str}loss{loss}_{add_to_path}results")

    df = pd.DataFrame({i: unjsonify(Path(mia_path, f"model_{i}.json")) for i in range(1, N_target_models+1)}).T
    df['model_id'] = df.index
    return df


######### Ploting code
def get_save_file(conf_type, N_shadow_models, target, loss, epsilon = "", plot_type = ""):
    return f"{conf_type}_{N_shadow_models}_{target}_loss_{loss}_epsilon_{epsilon}_{plot_type}.png"
def get_np_label(loss): return f"{methods['nonprivate']}"# ({losses[loss]} Loss)\n"
def get_dpsgd_label(loss, epsilon): return f"{methods['dpsgd']}"# ({losses[loss]} Loss, eps = {epsilon})\n"
def get_expm_label(loss, epsilon): return f"{methods['expm']}"# ({losses['l2']} Loss, eps = {epsilon})\n"

def hist(stat_array, labels = None, xlabel = "", title = "", legend_title = None, save_path = None, save_file = None, **plot_kwargs):

    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111, xlabel = xlabel, title = title)
    fig.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.hist(stat_array, label = labels, 
             **plot_kwargs)
    ax.set_ylabel("Number of Target Models")
    ax.grid(axis = "y", linestyle = ":", color = "grey", linewidth = "1.4")
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, fontsize = 8, loc='center left', 
                            bbox_to_anchor=(1, 0.5), title = legend_title, title_fontsize = 8)
    if save_path:
        print(f"Saving to {Path(save_path, save_file)}")
        fig.savefig(Path(save_path, save_file), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)

def mega_auc_plot(all_fpr, all_min_tpr, all_mean_tpr, all_max_tpr, epsilon, legend_title = None,  title = None, save_path = None, save_file = None, labels = [None], colors = [None], **plot_kwargs):
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111, title = title)
    for i in range(0, len(all_fpr)):
        fpr = all_fpr[i]
        min_tpr = all_min_tpr[i]
        mean_tpr = all_mean_tpr[i]
        max_tpr = all_max_tpr[i]
        label = labels[i]
        color = colors[i]
        plt.plot(fpr, mean_tpr, label = label, color = color, alpha = 1-i*.1, **plot_kwargs)
        plt.fill_between(fpr, min_tpr, max_tpr, color = light_colors[color], alpha = .5-.1*i)
    
    fig.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    ax.semilogx()
    ax.semilogy()
    ax.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.xlim(1e-5,1)
    plt.ylim(1e-5,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # put epslion in the bottom corner
    text = TextArea(f"$\\varepsilon$ = {epsilon}", textprops = {"fontsize": 8})
    box = AnchoredOffsetbox(child= text,
                        loc="lower right", frameon=True)
    
    box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(box)
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, title = legend_title, title_fontsize = 8, 
                    fontsize = 8, loc='center left', bbox_to_anchor=(1.05, .5))

    if save_path:
        print(f"Saving to {Path(save_path, save_file)}")
        fig.savefig(Path(save_path, save_file), bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close(fig)

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

def fig_shadow_confs(mia_results_path, conf_type, N_shadow_models, target, loss, 
                     add_to_path, data_inds = [], 
                     save_path = None, save_file = None):
    shadow_model_path = Path(mia_results_path, f"run_{conf_type}_{N_shadow_models}", "shadow_models",
                             f"{target}_loss_{loss}_{add_to_path}results", "mia_shadow_confs.json")
    shadow_df = pd.DataFrame.from_records(unjsonify(shadow_model_path))

    for ind in data_inds:
        confs_in = np.array(shadow_df[f"{ind}"]["confs_in"]).squeeze()
        min_in = np.min(confs_in)
        max_in =  np.max(confs_in)
        confs_out = np.array(shadow_df[f"{ind}"]["confs_out"]).squeeze()
        min_out = np.min(confs_out)
        max_out =  np.max(confs_out)
        min_range = np.min((min_in, min_out))
        max_range = np.max((max_in, max_out))
        
        fig = plt.figure(figsize=(9,6))
        gs = GridSpec(2, 2, figure=fig, width_ratios = [2, 1])
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])

        in_n, in_bins, _ = ax1.hist(confs_in, label = "confs when member", alpha = .5, color = 'blue')
        out_n, out_bins, _ = ax1.hist(confs_out, label = "confs when non-member", alpha = .5, color = 'green')
        x_axis = np.arange(min_range, max_range, .001)
        ax1.plot(x_axis, 
                 (max_range-min_range)/len(in_n) * sum(in_n) * norm.pdf(x_axis, shadow_df[f"{ind}"]["confs_in_mean"], shadow_df[f"{ind}"]["confs_in_std"]),
                 color = 'blue')
        ax1.plot(x_axis, 
                 (max_range-min_range)/len(out_n) * sum(out_n) *norm.pdf(x_axis, shadow_df[f"{ind}"]["confs_out_mean"], shadow_df[f"{ind}"]["confs_out_std"]),
                 color = 'green')
        handles, labels = ax1.get_legend_handles_labels()
        lgd = ax1.legend(handles, labels, fontsize = 8, loc='upper left', bbox_to_anchor=(.95, -0.1))

        # qq plot
        confs_in_norm = (np.array(shadow_df[f"{ind}"]["confs_in"]) - shadow_df[f"{ind}"]["confs_in_mean"])/shadow_df[f"{ind}"]["confs_in_std"]
        confs_out_norm = (np.array(shadow_df[f"{ind}"]["confs_out"]) - shadow_df[f"{ind}"]["confs_out_mean"])/shadow_df[f"{ind}"]["confs_out_std"]
        qqplot(confs_in_norm, ax = ax2, color = "blue")
        qqplot(confs_out_norm, ax = ax3, color = "green")

        if save_path:
            save_path.mkdir(exist_ok = True, parents=True)
            save_file = f"datapoint_{ind}_{conf_type}_{N_shadow_models}_{target}_{add_to_path}loss_{loss}.png"
            print(f"Saving to {Path(save_path, save_file)}")
            fig.savefig(Path(save_path, save_file), bbox_extra_artists = (lgd,), bbox_inches = 'tight')
        plt.close(fig)




if __name__ == '__main__':
    # stable results:
    conf_type = "stable"
    N_shadow_models = 1000
    N_target_models = 50
    loss = 'bce' # for nonprivate and dpsgd
    loss_str = "_bce"
    epsilons =  [.0001, .001, .01, .1, .5, 1, 2, 3.5, 5, 7, 10] #epsilons to test over 
    stat = 'ppv'

    for options in [(LR_MIA_RESULTS_FOLDER, LR_MIA_PLOTS_PATH, "", "mort_icu"),
                    (LR_MIA_RESULTS_FOLDER, LR_MIA_PLOTS_PATH, "", "los_3"),
                    (GRUD_MIA_RESULTS_FOLDER, GRUD_MIA_PLOTS_PATH, "nobn_", "mort_icu"),
                    (GRUD_MIA_RESULTS_FOLDER, GRUD_MIA_PLOTS_PATH, "nobn_", "los_3")]:
        mia_results_path, save_path, add_to_path, target = options
        
        # load attack results dataframes
        np_results_df = get_result_df(f"{conf_type}_{N_shadow_models}", "nonprivate", N_target_models, target, loss_str, "mia_nonprivate_baseline.json", mia_results_path, add_to_path)
        np_audit_df = get_lower_eps(audit_df(np_results_df))
        
        np_lira_df = attack_stats_df(np_results_df)
        np_target_models_df = get_target_model_df(f"{conf_type}_{N_shadow_models}", "nonprivate", N_target_models, target, loss_str, mia_results_path, add_to_path)
       
        dpsgd_results = {epsilon : get_result_df(f"{conf_type}_{N_shadow_models}", "dpsgd", N_target_models, target, loss_str, f"mia_dpsgd_epsilon_{epsilon}.json", mia_results_path, add_to_path) for epsilon in epsilons}
        dpsgd_results_df = pd.concat(dpsgd_results)
        dpsgd_results_df = dpsgd_results_df.reset_index().drop(columns = 'level_1')
        dpsgd_results_df = dpsgd_results_df.rename(columns = {'level_0': 'epsilon'})
        dpsgd_audit_df = get_lower_eps(audit_df(dpsgd_results_df))
        dpsgd_lira_df = pd.concat({epsilon : attack_stats_df(dpsgd_results[epsilon]) for epsilon in epsilons})
        dpsgd_lira_df = dpsgd_lira_df.reset_index().drop(columns = 'level_1')
        dpsgd_lira_df = dpsgd_lira_df.rename(columns = {'level_0': 'epsilon'})
        dpsgd_target_models_df = pd.concat({epsilon : get_target_model_df(f"{conf_type}_{N_shadow_models}", "dpsgd", N_target_models, target, loss_str, mia_results_path, add_to_path, epsilon_str = f"epsilon_{epsilon}_") for epsilon in epsilons})
        dpsgd_target_models_df = dpsgd_target_models_df.reset_index().drop(columns = 'level_1')
        dpsgd_target_models_df = dpsgd_target_models_df.rename(columns = {'level_0': 'epsilon'})

        if mia_results_path == LR_MIA_RESULTS_FOLDER:
            expm_results = {epsilon : get_result_df(f"{conf_type}_{N_shadow_models}", "expm", N_target_models, target, "_l2", f"mia_expm_epsilon_{epsilon}.json", mia_results_path, "") for epsilon in epsilons}
        else:
            expm_results = {epsilon : get_result_df(f"{conf_type}_{N_shadow_models}", "expm", N_target_models, target, "", f"mia_expm_epsilon_{epsilon}.json", mia_results_path, "") for epsilon in epsilons}
        expm_results_df = pd.concat(expm_results)
        expm_results_df = expm_results_df.reset_index().drop(columns = 'level_1')
        expm_results_df = expm_results_df.rename(columns = {'level_0': 'epsilon'})
        expm_audit_df = get_lower_eps(audit_df(expm_results_df))
        expm_lira_df = pd.concat({epsilon : attack_stats_df(expm_results[epsilon]) for epsilon in epsilons})
        expm_lira_df = expm_lira_df.reset_index().drop(columns = 'level_1')
        expm_lira_df = expm_lira_df.rename(columns = {'level_0': 'epsilon'})
        expm_target_models_df = pd.concat({epsilon : get_target_model_df(f"{conf_type}_{N_shadow_models}", "expm", N_target_models, target, "l2", mia_results_path, "", epsilon_str = f"epsilon_{epsilon}_") for epsilon in epsilons})
        expm_target_models_df = expm_target_models_df.reset_index().drop(columns = ['level_0', 'level_1'])

        # save auditing results
        audit_save_path = Path(mia_results_path, f"run_{conf_type}_{N_shadow_models}", "privacy_audit")
        audit_save_path.mkdir(exist_ok=True, parents=True)
        jsonify({"non_private": np_audit_df.to_dict(), 
                 "dpsgd": dpsgd_audit_df.to_dict(),
                 "expm": expm_audit_df.to_dict()}, 
                Path(audit_save_path, f"{target}_loss_{loss}_{N_shadow_models}.json"))


        ########## Begin plotting
        def save_file(epsilon, plot_type): return get_save_file(conf_type, N_shadow_models, target, loss, epsilon = epsilon, plot_type = plot_type)

        # creates QQ plots
        fig_shadow_confs(mia_results_path, conf_type, N_shadow_models, target, loss, 
                         add_to_path, data_inds = [4393, 13280], 
                         save_path = Path(save_path, "shadow_models"))

    
        for epsilon in epsilons:
            mega_auc_plot([np_lira_df.groupby('fpr')['tpr'].max().index.to_numpy(),
                           dpsgd_lira_df.groupby(['fpr', 'epsilon'])['tpr'].max().reset_index().query(f'epsilon == {epsilon}')['fpr'].to_numpy(),
                           expm_lira_df.groupby(['fpr', 'epsilon'])['tpr'].max().reset_index().query(f'epsilon == {epsilon}')['fpr'].to_numpy()
                     ],
                     [np_lira_df.groupby('fpr')['tpr'].quantile(.05).to_numpy(),
                      dpsgd_lira_df.groupby(['fpr', 'epsilon'])['tpr'].quantile(.05).reset_index().query(f'epsilon == {epsilon}')['tpr'].to_numpy(),
                      expm_lira_df.groupby(['fpr', 'epsilon'])['tpr'].quantile(.05).reset_index().query(f'epsilon == {epsilon}')['tpr'].to_numpy()
                     ],
                     [np_lira_df.groupby('fpr')['tpr'].median().to_numpy(),
                      dpsgd_lira_df.groupby(['fpr', 'epsilon'])['tpr'].median().reset_index().query(f'epsilon == {epsilon}')['tpr'].to_numpy(),
                      expm_lira_df.groupby(['fpr', 'epsilon'])['tpr'].median().reset_index().query(f'epsilon == {epsilon}')['tpr'].to_numpy()
                     ],
                     [np_lira_df.groupby('fpr')['tpr'].quantile(.95).to_numpy(),
                      dpsgd_lira_df.groupby(['fpr', 'epsilon'])['tpr'].quantile(.95).reset_index().query(f'epsilon == {epsilon}')['tpr'].to_numpy(),
                      expm_lira_df.groupby(['fpr', 'epsilon'])['tpr'].quantile(.95).reset_index().query(f'epsilon == {epsilon}')['tpr'].to_numpy()
                     ], epsilon,
                     labels = [f'{get_np_label(loss)}',# AUC {round(np_lira_df.auc.median(), 4)}', 
                               f"{get_dpsgd_label(loss, epsilon)}",# AUC {round(dpsgd_lira_df.groupby('epsilon').auc.median().reset_index().query(f'epsilon == {epsilon}')['auc'].median(), 4)}", 
                               f"{get_expm_label(loss, epsilon)}"],# AUC {round(expm_lira_df.groupby('epsilon').auc.median().reset_index().query(f'epsilon == {epsilon}')['auc'].median(), 4)}",],
    #                           f'{get_expm_label(loss, epsilon)} AUC {round(expm_lira_df.auc.mean(), 4)}'],
                     title = f"Median ROC Curve over {N_target_models} Models", colors = [non_pr_color, dpsgd_color, expm_color],
                     legend_title = targets[target],
                     save_file = save_file(epsilon, f"auc"), save_path = save_path)

            for stat in ['ppv', 'npv']:
                hist([np_lira_df.groupby(['model_id'])[stat].max().to_numpy(),
                      dpsgd_lira_df.groupby(['model_id', 'epsilon'])[stat].max().reset_index().query(f'epsilon == {epsilon}')[stat].to_numpy(),
                      expm_lira_df.groupby(['model_id', 'epsilon'])[stat].max().reset_index().query(f'epsilon == {epsilon}')[stat].to_numpy(),
                      ], 
                      labels = [get_np_label(loss), get_dpsgd_label(loss, epsilon), get_expm_label(loss, epsilon)],
                     title = f"Maximum Possible {stats[stat]} for Attack", xlabel = stats[stat],
                     range = (.49, 1.01), bins = 6,
                     color = [non_pr_color, dpsgd_color, expm_color], legend_title = targets[target],
                     save_file = save_file(epsilon, f"max_{stat}_hist"), save_path = save_path)

            for (stat, thresh_stat, t) in [('mem_correct_pred', 'ppv', .99),
                                           ('nonmem_correct_pred', 'npv', .99)]:   
                hist([np_lira_df.groupby(['model_id', thresh_stat])[stat].max().reset_index().query(f'{thresh_stat} > {t}').groupby('model_id')[stat].max(),
                      dpsgd_lira_df.groupby(['model_id', 'epsilon', thresh_stat])[stat].max().reset_index().query(f'{thresh_stat} > {t} & epsilon == {epsilon}').groupby('model_id')[stat].max(),
                      expm_lira_df.groupby(['model_id', 'epsilon', thresh_stat])[stat].max().reset_index().query(f'{thresh_stat} > {t} & epsilon == {epsilon}').groupby('model_id')[stat].max()],
                     labels = [get_np_label(loss), get_dpsgd_label(loss, epsilon), get_expm_label(loss, epsilon)],
                     title = f"{stats[stat]} when {stats_acr[thresh_stat]} > {t}", xlabel = stats[stat],
                     color = [non_pr_color, dpsgd_color, expm_color], legend_title = targets[target],
                     save_file = save_file(epsilon, f"{stat}_hist"), save_path = save_path)





