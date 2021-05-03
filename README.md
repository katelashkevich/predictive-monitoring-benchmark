# Business Process Mining. Homework 4: Predictive Process Monitoring
Solution to the home task 4 of the Business Process Mining course (University of Tartu)

Original codebase [by Irene Teinemaa](https://github.com/irhete/predictive-monitoring-benchmark)

The original codebase was updated to support a new dataset [turnaround_anon_sla.csv](http://kodu.ut.ee/~chavez85/pm_course/data/turnaround_anon_sla.csv), and use **n-grams** instead of **prefixes** in the original approach.

### Repository structure
- `HW4.ipynb` - **main Jupyter Notebook file containing homework task solution and explanations of original codebase changes**
- `input` folder - contains raw turnaround_anon_sla.csv dataset
- `labeled_logs_csv_processed` folder - contains labeled dataset with WIP column (Tasks 1 and 2). Used as an input file for `optimize_params.py`
- `bucketers` and `transformers` folders - copied from the original codebase without changes
- `BucketFactory.py` and `EncoderFactory.py` - used without changes (were moved from `experiments` folder in the original repo to the repo root)
- `DatasetManager.py` - was moved from `experiments` folder in the original repo to the repo root. Includes new `generate_ngram_data()` and `get_ngram_shifts()` methods to allow using n-grams instead of log prefixes
- `dataset_confs.py` - was moved from `experiments` folder in the original repo to the repo root. Includes new configuration for `turnaround_labeled` dataset
- `optimize_params.py` - was moved from `experiments` folder in the original repo to the repo root. Uses `dataset_manager.generate_ngram_data()` method istead of `dataset_manager.generate_prefix_data()`
- `experiments.py` - was moved from `experiments` folder in the original repo to the repo root. Uses `dataset_manager.get_ngram_shifts()` method istead of `dataset_manager.get_prefix_lengths()`, and `dataset_manager.generate_ngram_data()` method istead of `dataset_manager.generate_prefix_data()`
- `output` folder - contains outcomes of executinf `optimize_params.py` and `experiments.py`