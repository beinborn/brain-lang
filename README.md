# Encoding fMRI Data
This repository contains code for evaluating language--brain encoding experiments. 


# Reader
We provide readers for four datasets:
* The [Words Data](http://www.cs.cmu.edu/~fmri/science2008/data.html) by Mitchell et al. (2008)
* The [Alice Data](https://sites.lsa.umich.edu/cnllab/2016/06/11/data-sharing-fmri-timecourses-story-listening/) by Brennan et al. (2016) 
* The  [Harry Potter Data](https://drive.google.com/file/d/0By_8Ci8eoDI4Q3NwUEFPRExIeG8/view) by Wehbe et al. (2018)
* The [Stories Data](https://onlinelibrary.wiley.com/doi/abs/10.1002/hbm.23814) by Dehghani et al. (2017)


# Language Models
We provide a class to add a language model and implementations for an Elmo reader (Peters et al. (2018)) and a random language model 

# Mapping Model
* The mapping model is standard ridge regression


# Evaluation
We provide code for three common evaluation procedures:
* pairwise evaluation
* voxel-wise evaluation
* representational similarity analysis 

# Experiments
There are two experimental pipelines, one for isolated stimuli (e.g., single words) and one for continuous stimuli (e.g., a book chapter). 
You can run them as follows: 
* python3 continuous_stimuli_experiments.py
* python3 isolated_stimuli_experiments.py

If you want to better understand the data structure, have a look at test_readers.py

## Requirements:
* Numpy
* Sklearn
* Allennlp (for Elmo)
* Spacy for tokenization: python -m spacy download en_core_web_lg
* Pandas, matplotlib, seaborn and nilearn in case you want to plot the results. 

