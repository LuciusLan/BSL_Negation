# Negation Scope Refinement via Boundary Shift Loss
------

This repository contains codes for my paper:
> [Yin Wu and Aixin Sun. 2023. Negation Scope Refinement via Boundary Shift Loss. In Findings of the Association for Computational Linguistics: ACL 2023. Association for Computational Linguistics.](https://aclanthology.org/2023.findings-acl.379/) 

Requirements:

python: 3.x

pytorch: 1.5+

gensim: 3.x



The data pre-processing part (processor.py and data.py) were adapted based on Aditya and Suraj's [Transformers-For-Negation-and-Speculation](https://github.com/adityak6798/Transformers-For-Negation-and-Speculation), but added my personal label augmentation options and changed the dataloader structure.


You will need to store the downloaded data (please collect from the corresponding dataset owners) into /Data/ folder, in the forms listed in params.py (data_path), in order to run the experiments.

Various settings are listed in params.py, you may try to adjust different hyperparameters.

Sorry for the codes contains a lot of line irrelevant to the paper (which were used for previous exploration of solutions)
