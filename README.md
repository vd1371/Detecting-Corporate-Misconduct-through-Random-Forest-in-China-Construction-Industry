# Detecting-Corporate-Misconduct-through-Random-Forest-in-China-Construction-Industry

///-----------------------------------/// / Vahid Asghari / / / /---------------------------------------/

Audience: Anyone with beginner level of python programming skills, and looking for a simple code for random forest to cunduct regression analysis

This code was originally used in the paper "Detecting Corporate Misconduct through Random Forest in China’s Construction Industry", by Ran Wang , Vahid Asghari , Shu-Chien Hsu , Chia-Jung Lee , and Jieh-Haur Chen. If you are using this code, please cite this paper and following libraries (scikit-learn)

For using this code, please follow these steps:

1- After preprocessing your data (such as feature scaling (except the output variable), removing outliers,...), save the dataset as a "csv" file.
iris is a freely available dataset for test in this repository.
The first column of the file should be the index or date and the last column of the dataset must be the output variable.

If you want the code to slive the data for training and testing, please set auto_shuffle = True
Else, slice your data into to datasets following the above format and rename them exactly like the example solved in this repository

2- Open 'RandomForest.py' using any IDE and Fill in the necessary settings:

2-1: auto_shuffle: turns on shuffling the data before fitting
2-2: should_cross_val: whether using k-fold cross validation or not


Most of the settings are self explanatory and can be found in the paper
For more information regarding the hyperparameters, please refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


3- fit():
fits the random forest based on the settings

5- tune(grid):
gets grid and tries to find the best set of hyperparameters by search in this grid

5-1: shoud_random_search: whether search for the best set of hyperparameters randomly or searches for all possible combinations
5-2: n_iter: number of iteration in random search for tuning
5-3: "tune" is designed when the auto_shuffle is True. Otherwise, it does not work. You need to pass the whole dataset to the code.

6- Results can be found in the report folder with the name of your dataset