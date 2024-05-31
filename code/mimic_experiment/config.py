import sys 
from pathlib import Path 

losses = ['l2', 'bce'] # which loss functions to test for DPSGD , nonprivate. (expm requires l2 currently)
# membership inference uses bce.
targets = ['mort_icu', 'los_3'] # which prediction targets to use for MIMIC 3 ICU data. 

DATA_FOLDER = Path('data/mimic')
RESULTS_FOLDER = Path('results/mimic')

## LR folders
LR_BASELINE_RESULTS_FOLDER = Path('results/mimic/lr_nonprivate_baseline')
LR_DPSGD_RESULTS_FOLDER = Path('results/mimic/lr_dpsgd')
LR_EXPM_RESULTS_FOLDER = Path('results/mimic/lr_expm')
LR_PLOTS_PATH= Path('results/mimic/lr_plots')

## now make these folders:
LR_BASELINE_RESULTS_FOLDER.mkdir(exist_ok = True, parents=True)
LR_DPSGD_RESULTS_FOLDER.mkdir(exist_ok=True)
LR_EXPM_RESULTS_FOLDER.mkdir(exist_ok = True)
LR_PLOTS_PATH.mkdir(exist_ok=True)

## LR MIA folders
LR_MIA_RESULTS_FOLDER = Path('results/mimic/lr_mia')
LR_MIA_PLOTS_PATH= Path('results/mimic/lr_mia_plots')

## now make these folders:
LR_MIA_RESULTS_FOLDER.mkdir(exist_ok = True, parents=True)
LR_MIA_PLOTS_PATH.mkdir(exist_ok=True)

## GRUD folders
GRUD_BASELINE_RESULTS_FOLDER = Path('results/mimic/grud_nonprivate_baseline')
GRUD_DPSGD_RESULTS_FOLDER = Path('results/mimic/grud_dpsgd')
GRUD_EXPM_RESULTS_FOLDER = Path('results/mimic/grud_expm')
GRUD_PLOTS_PATH= Path('results/mimic/grud_plots')

# now make these folders: 
GRUD_BASELINE_RESULTS_FOLDER.mkdir(exist_ok = True, parents=True)
GRUD_DPSGD_RESULTS_FOLDER.mkdir(exist_ok=True)
GRUD_EXPM_RESULTS_FOLDER.mkdir(exist_ok=True)
GRUD_PLOTS_PATH.mkdir(exist_ok=True)

## GRUD MIA folders
GRUD_MIA_RESULTS_FOLDER = Path('results/mimic/grud_mia')
GRUD_MIA_PLOTS_PATH= Path('results/mimic/grud_mia_plots')

## now make these folders:
GRUD_MIA_RESULTS_FOLDER.mkdir(exist_ok = True, parents=True)
GRUD_MIA_PLOTS_PATH.mkdir(exist_ok=True)


