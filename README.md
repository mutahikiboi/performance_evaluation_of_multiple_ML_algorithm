# Performance_evaluation_of_multiple_ML_algorithms
Benchmarking ML Algorithms with mlr3 and mlr3pipelines on Penguins Dataset
This project evaluates the performance of multiple machine learning algorithms ("classif.rpart", "classif.ranger", "classif.svm", and "classif.featureless") on the penguins dataset using the mlr3 ecosystem. Key steps include:

Data Preprocessing: Handling missing values via mlr3pipelines (imputation, encoding).

Model Training: Comparing decision trees, random forests, and SVMs with automated workflows.

Benchmarking: Rigorous evaluation using 10-fold cross-validation with 10 repetitions.

Visualization: Analyzing results to identify the best-performing model.

##Highlights:
🔹 Pipeline construction with GraphLearner for seamless preprocessing + training.
🔹 Fair benchmarking with instantiated resampling to ensure reproducibility.
🔹 Code templates for extending to other datasets/tasks.

#License
This project is licensed under the MIT License - see the LICENSE file for details.

