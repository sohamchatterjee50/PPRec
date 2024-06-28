# PP-Rec

PP-Rec Source code

The documented version of the code is present in https://github.com/sohamchatterjee50/PPRec/tree/main/DocumentedCode. Some of the utility methods are called from https://github.com/sohamchatterjee50/PPRec/tree/main/Baseline_Models/ebnerd-benchmark-main

Steps to reproduce Baseline experiemnts:
1. First go to folder Baseline_Models/ebnerd-benchmark-main
2. To install the dependencies, run the command "pip install ."
3. Now go to foler DocumentedCode.
4. To install the dependecnies for this project, run the command again "pip install .". Note that we run the command "pip install ." at 2 different places to install 2 different packages.
5. Go to the notebook DocumentedCode/train.ipynb.
6. The notebook train.ipynb is well documented with all the modules, training and evaluation pipelines.
7. Run the entire notebook to run the training pipeline with simultaneous logging of checkpoints as well as fetch the evaluation metrics.


Here I have pre-processed the demo datasets and related dictionary mappings and stored them inside DocumentedCode/demo_processed.
For preprocessing documentation and other details, run the notebook DocumentedCode/preprocess.ipynb. This notebook is also documented with all the necessary signals added at different steps.

For ablation studies,
1. To remove Time-aware News Popularity Predictor, comment out the corresponding module from PPRec class.
2. To remove recency or CTR, comment out the corresponding features inside Time-aware News Popularity Predictor class.
3. To combine NER, topics and category info, preprocess the dataset and take the mean embedding inside KnowledgeAwareNewsEncoder module.

Extensions:
1. For user profiling, preprocess the dataset and the dataloader to add the user meta data related signals.
2. Apply  torch.nn.MultiheadAttention between the current user embedding with clicked historical news info and the user profiel embedding,ie- apply cross attention.
3. For topic level diversity, sample the dataset and follow the procedure where for every session of 5 articles, one is the clikec article, 2 negative articles are from the same topic and the other 2 from a different topic as the clicked article.

Extension-Multi Task:
1. First go to folder OurCode/src/Ad_kai/ebnerd-benchmark
2. To install the dependencies, run the command "pip install ."
3. Go to OurCode
4. run "python train_kai.py" to run multi task experiment on demo dataset. (To test on small dataset change the first two lines DATASPLIT=ebnerd_small testsplit=val_SMALL.parquet)

