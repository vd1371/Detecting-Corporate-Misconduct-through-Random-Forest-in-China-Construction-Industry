import os, sys
from Reporter import *

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

class RandomForest(Report):
    
    def __init__(self,	name = None,
                        split_size = 0.2,
                        auto_shuffle = True,
                        k = 5,
                        num_top_features = 10,
                        imbalanced = False):
        
        super(RandomForest, self).__init__(name)
        
        self.num_top_features = num_top_features
        self.k = k
        self.auto_shuffle = auto_shuffle

        if auto_shuffle:
        	df = pd.read_csv(name+".csv", index_col = 0)
        	self.X = df.iloc[:,:-1]
        	self.Y = df.iloc[:,-1]

        	self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size = split_size, shuffle = True)

        else:
        	try:
        		df_train = pd.read_csv(name+"_train.csv", index_col = 0)
        	except FileNotFoundError:
        		print ("There is a problem is loading the train file. Please rename it like filename_train.csv")
        	try:
        		df_test = pd.read_csv(name+"_test.csv", index_col = 0)
        	except FileNotFoundError:
        		print ("There is a problem is loading the train file. Please rename it like filename_test.csv")

        	self.X_train = df_train.iloc[:, :-1]
        	self.Y_train = df_train.iloc[:, -1]
        	self.X_test = df_test.iloc[:, :-1]
        	self.Y_test = df_test.iloc[:, -1]

        self.dates_train = self.X_train.index
        self.dates_test = self.X_test.index

        if imbalanced:
        	smt = SMOTE()
        	self.X_train, self.Y_train = smt.fit_sample(self.X_train, self.Y_train)
        	self.dates_train = ['SMOTE'+str(i) for i in range(len(self.X_train))]

        
    def initialize(self, n_estimators=10000,
    					max_depth=None,
    					min_samples_split=2,
    					min_samples_leaf=1, 
                  		max_features='auto',
                  		bootstrap=True,
                  		n_jobs=-1,
                  		verbose=1,
                  		should_cross_val = True):

    	# For further information regarding RandomForest library...
    	# ... please visit: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.should_cross_val = should_cross_val
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Initializing the model
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
        			max_depth=self.max_depth,
        			min_samples_split=self.min_samples_split,
        			min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    bootstrap=self.bootstrap,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose)
    
    @timeit
    def fit(self):
        
        # Logging the initialization of the fitting
        self.log.info(f'--- Random Forest with {self.n_estimators} estimators is about to fit')
         
        self.model.fit(self.X_train, self.Y_train)
        print (f"Random Forest is fitted")
            
        if self.should_cross_val and self.auto_shuffle:
            scores = cross_val_score(self.model, self.X, self.Y, cv=self.k, verbose=0)
            self.log.info(f"---- Cross validation with {self.k} groups----\n\nThe results on each split" + str(scores)+"\n")
            self.log.info(f"The average of the cross validation is {np.mean(scores):.2f}\n")
            
            print (f"Cross validation is done for Random Forest. Score: {np.mean(scores):.2f}")
    
        self.evaluate_classification(self.Y_train, self.model.predict(self.X_train), self.dates_train, 'RF-OnTrain')
        self.evaluate_classification(self.Y_test, self.model.predict(self.X_test), self.dates_test, 'RF-OnTest')
        
        # Plotting the Importances
        feature_importances_ = {}
        for i in range(len(self.model.feature_importances_)):
            feature_importances_[self.X_test.columns[i]] = self.model.feature_importances_[i]
        
        self.report_feature_importance(feature_importances_, self.num_top_features, label = 'RF')
        
    @timeit
    def tune(self, grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                'max_features': ['auto', 'sqrt'],
                                'max_depth': [int(x) for x in np.linspace(5, 20, num = 15)] + [None],
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'bootstrap': [True, False]},
                                should_random_search = False,
                                n_iter = 1):
            
        self.log.info(f'------ Random Forest is going to be Tunneds -----')
        if should_random_search:
            searched_model = RandomizedSearchCV(estimator = self.model, param_distributions = grid, n_iter = n_iter, cv = self.k, verbose=2, n_jobs = self.n_jobs)
        else:
            searched_model = GridSearchCV(estimator = self.model, param_distributions = grid, cv = self.k, verbose=2, n_jobs = self.n_jobs)


        self.model = searched_model
        self.model.fit(self.X, self.Y)
        
        self.log.info(f"\n\nBest params:\n{pprint.pformat(searched_model.best_params_)}\n")
        self.log.info(f"\n\nBest score: {-1*searched_model.best_score_:0.4f}\n\n")
        print (f'The best model MSE is: {-1*searched_model.best_score_:0.4f}')

    def save(self):
    	joblib.dump(self.model, self.directory + f"/RF.pkl")
        
        
def run_me():
    file_name = 'MisCond'

    model = RandomForest(file_name, split_size=0.2, auto_shuffle=False, k=5, num_top_features = 20, imbalanced = True)
    
    model.initialize(n_estimators = 100,
                 max_depth=5,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='auto',
                 should_cross_val=False)

    model.fit()

    # For finding the best hyperparameters
    # model.tune(grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)],
    #                      'max_features': ['auto', 'sqrt'],
    #                      'max_depth': [int(x) for x in np.linspace(3, 20, num = 16)] + [None],
    #                      'min_samples_split': [val for val in np.linspace(start = 0.1, stop = 0.9, num = 9)],
    #                      'min_samples_leaf': [val for val in np.linspace(start = 0.1, stop = 0.5, num = 5)],
    #                      'bootstrap': [True, False]},
    #                      should_random_search = True,
    #                      n_iter = 1)
    

if __name__ == "__main__":
    run_me()

        