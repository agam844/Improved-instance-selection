# Improved-instance-selection
Active Learning's central concept lies in its capacity to efficiently manage unlabelled data by strategically selecting samples for labelling. 
This process proves especially advantageous due to the time and expertise required for labelling. Moreover, Active Learning's benefits can be extended to labelled datasets as well, 
where it aids in choosing informative subsets for training. This is valuable given the computational costs associated with training on large datasets, optimizing resource utilization.

This project focuses on refining the process of selecting training data subsets for fine-tuning a machine learning model in toxicity classification through active learning.
The study also demonstrates that enhancing this subset selection task is achievable by combining the prediction uncertainty from a large language Model with that of a smaller language model. 
This approach effectively improves the efficiency of the selection process, thereby advancing the overall performance of the model.

Toxicity classification dataset taken from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
