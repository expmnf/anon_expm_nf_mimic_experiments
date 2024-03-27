# ExpM NF MIMIC3 results
This code allows reproduction of results in the paper 

"Are Normalizing Flows the Key to Unlocking the Exponential Mechanism? A Path through the Accuracy-Privacy Ceiling Constraining Differentially Private ML"

The code here provides implementation of our experiments where we test if we can training models with ExpM+NF alongside and in comparison with DPSGD and non-private training.  


(See data set up instructions for citations needed when using MIMIC-III data and MIMIC_Extract's pipeline. )

---

## Repository Setup 
We used git-lfs for results files. Install git-lfs using your favorite package manager. Run `git lfs install` to initiate it. 

We used pyenv for managing different versions of python and venv for our python virtual environment. 

Quick background and useful commands appear at the bottom of this readme. 

The setup follows reference: https://www.freecodecamp.org/news/manage-multiple-python-versions-and-virtual-environments-venv-pyenv-pyvenv-a29fb00c296f/ 

Steps: 
1. Install python 3.9.14 via pyenv using `pyenv install 3.9.14`)

2. Set pyenv envrionment to python 3.9.14 using  `pyenv local 3.9.14`    

3.  Initialize virtual envrionment in the .venv folder `python3 -m venv .venv`. Now, in the new folder `.venv/bin` should be a copy of python3.9 

    Note that the `.venv` folder is gitignored and should not ship with the repo. 

3. Start virtual envrionment: `source .venv/bin/activate` 

    At this point running `which pip` and `which python` should produce a path to the `pip` and `python` instances in `.venv/bin/`. 

    Running `python --version` should produce 3.9.14. 

    VS Code and Jupyter users may need to point their environment to the right Python interpreter. (`.vinv/bin/python`)

4. We shipped our `pip freeze` output as the `requirements_*.txt` files file, so one can (try to) install all packages with `pip install -r requirements.txt`. If that is problematic, one can try `cat requirements.txt | xargs -n 1 pip install` following https://stackoverflow.com/questions/22250483/stop-pip-from-failing-on-single-package-when-installing-with-requirements-txt (For some reason to install scikit requires: `python3 -m pip install scikit-learn`)

---

## Folder structure is 
  - /.venv/ folder (.gitignored) holds pyvenv copies of python etc. Created  by you using the setup above. 

  - /data/* (gitignored) holds the mimic3 data, created by you. Below is what it should have after downloading mimic3 data and preprocessing it. 
    - /mimic/ 
        - all_hourly_data.h5,  must be downloaded. See notebooks/mimic_preprocessing/README.md for info/directions
        - lvl2_l_inf_normalized.h5 l_infinity normalized mimic3 data, note! not pivoted on hours yet
        - lvl2_z_normalized.h5 z-score normalized mimic3 data. note! not pivoted on hours yet
        - Ys_mimic_extract.csv - First three columns are multi-indices for the patient, remaining columns are the targets. **Use `Ysr = pd.read_csv('data/mimic/Ys_mimic_extract.csv', index_col=[0,1,2])`** when you read it in! 

  - /code/ folder holds python scripts that in general process /data into /results 

  - /notebooks/ folder for jupyter notebooks

  - /results/ folder with results from code/ 

---

## Getting MIMIC-III dataset ready for experiments
We use the MIMIC-III data version 1.4 (https://physionet.org/content/mimiciii/1.4/) and MIMIC-III data processing pipeline called MIMIC-Extract of Wang et al. (https://github.com/MLforHealth/MIMIC_Extract). Please cite their papers when using their data and code, see subsection below with their citation requests. 

The concise todos are itemized here (with in depth details to follow): 
1.  First go to https://physionet.org/ create a login and required trainings for accessing the MIMIC-III
2.  Create `/data/mimic/` folder in this repo (/data already gitignored)
3.  Download the data. While the direct URL for the dataset is page is https://physionet.org/content/mimiciii/1.4/, once Physionet verified+trainings are complete, you should have access to  [this link]( https://console.cloud.google.com/storage/browser/mimic_extract). Download the `all_hourly_data.h5` and put it in `/data/mimic` 
    Some notes:
	- in the (MIMIC-Extract repo)[https://github.com/MLforHealth/MIMIC_Extract#pre-processed-output] it says 
		> If you simply wish to use the output of this pipeline in your own research, a preprocessed version with default parameters is available via gcp, [here](https://console.cloud.google.com/storage/browser/mimic_extract).
	- this is the "level_2" data in the mimic-extract and chasing long tails paper it seems. 
	- the "raw" data is w/o the second layer aggregation. We believe and it does not appear in this google cloud account. One could conceivably rerun the pipeline from mimic-extract to make it. In that case, the "full" mimic-3 data would be supplied by physionet. At the bottom of the MIMIC-III v1.4 page are the download options. Register your google email as an email in your physionet account and verify email. 
4. Run either `/code/mimic_experiment/mimic_extract_preprocess.py` or the equivalent notebook `/notebooks/mimic_extract_preprocess.ipynb` (in this repo). It will read in the `/data/mimic/all_hourly_data.h5` data. It will create two `lvl_*.h5` files in the same location. 
	- Discrepancies with Suriyakumar et al. are that we 
    	- have 104 features, but they claim to have 78
    	- have 89.1% missingness before imputation but they claim to have 78%
	- Notably the number of rows matches exactly. 
Discussion of these directions from MIMIC_Extract repo is [here](https://github.com/MLforHealth/MIMIC_Extract/issues/46#issuecomment-767023459)


### Citations for using MIMIC-III: 
When using the mimic3 data resource, please cite: 
- Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet. https://doi.org/10.13026/C2XW26.

- Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.

- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

When using MIMIC_Extract pipeline please cite: 
- Shirly Wang, Matthew B. A. McDermott, Geeticka Chauhan, Michael C. Hughes, Tristan Naumann, and Marzyeh Ghassemi. MIMIC-Extract: A Data Extraction, Preprocessing, and Representation  Pipeline for MIMIC-III. arXiv:1907.08322. 

### Some notes on our MIMIC-3 data: 
Goal was to reproduce benchmarks in the following paper:
> ["Chasing your long tails" Suriyakumar 2021](https://dl.acm.org/doi/pdf/10.1145/3442188.3445934)
and to create approapriate dataset to build on their work. The Supplementary Information for the paper describes their data processing details.  
In brief, the benchmark paper uses the mimic-extract pipeline defaults:
1. patient first ICU visit
2. all patients older than 15 years of age
3. _Except:_ ICU stay > 36 hours (mimic-extract: > 12 hours)
They arrive with a cohort of 21,877 unique ICU stays. (we do too)
Discrepancies with Suriyakumar et al. are that we 
    	- have 104 features, but they claim to have 78
    	- have 89.1% missingness before imputation but they claim to have 78%
#### Extra pre-process steps needed
1. Filter for ICU stay > 36 hours (as noted above)
2. Filter for patients missing `diagnosis_at_admission`
> NOTE:  There was 1 patient with missing record for `diagnosis_at_admission`.
After dropping them, the total cohort selection = 21,877

#### MIMIC-Extract pipeline
[Github](https://github.com/MLforHealth/MIMIC_Extract)

#### Pre-processed Output
A preprocessed version with default parameters is available via gcp,
[here](https://console.cloud.google.com/storage/browser/mimic_extract).

#### Related Notebooks in MIMIC-extract repo:
[Jupyter notebooks for tasks](https://github.com/MLforHealth/MIMIC_Extract/tree/master/notebooks)

---

## Running the Experiments 
### Accuracy and privacy results: 
To create accuracy and privacy results for each training <type> (nonprivate/dpsgd/expm) and each <model> (logistic_regression/grud) experiments, one will use `code/mimic_experiment/<model>/run_<type>.py`, e.g., for dpsgd training on logistic_regression use `code/mimic_experiment/logistic_regression/run_dpsgd.py`. 
- We ran this three times, two hyperparameter searching runs and a third final run. 
    1. first run uses `h_pass = "first", use_full_train = False` (broad hyperparmeter search w/ validation on dev set.)
    2. second run uses `h_pass = "refined", use_full_train = False` (refined hyperparms based results of first run w/ validation on dev set.)
    3. final results run uses `h_pass = "refined", use_full_train = False` (best hyperparms previously observed are use, now with validation on the test set.) 
- To set up the script:
    - lines ~15 are the GPU/CPU setup and will have to be configured for your enviornment (`gpu_names, device_ids, use_gpu = setup_devices(devices)`)
    - lines ~17 are parameters to set as follows: 
      - `use_full_train = False` will use train/dev splits (for all hyperparm search runs). `use_full_train = True` will use train+dev/ test (for final results only) 
      - `h_pass`: tells which hyperparmeters to use in the code/<model>/hyperparemeter_sets.py.

One must have results from `h_pass = refined` to run `h_pass = final` as it uses the best found hyperparms in the `refined` run. 

Output will appear in results folder according to the path in line ~215: `RESULTS_FOLDER = Path(FOLDER, f"{h_pass}{run}")` which is printed upon writing. 

Hyperparmeter search spaces we used are in the `hyperparm_sets.py` scripts. 

### Timing benchmark results 
To create the timing benchmarking results for each  training <type> (nonprivate/dpsgd/expm) an each <model> (logistic_regression/grud) experiments, one will run `code/mimic_experiment/<model>/timing_benchmark_<type>.py`, e.g., for dpsgd training on logistic_regression we would open `code/mimic_experiment/logistic_regression/timing_benchmark_dpsgd.py`. 

These scripts are set up similarly to the `run_<type>.py` scripts. 
We used the full dataset for timing benchmarks, i.e. `use_full_train = True`. 
Set `h_pass = benchmarks`. 

See Pytorch Timer source code if desired: 
- https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/utils/timer.py
- https://github.com/pytorch/pytorch/blob/main/torch/utils/benchmark/utils/common.py#L74 

### Plotting results: 
For each <model> (logistic regression /grud) there is a plotting code file that can be run to create the result plots, namely `/code/mimic_experiment/<model>/plot_results.py`

Happy reproducing! 
