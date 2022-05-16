import pickle
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold, GroupShuffleSplit, cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from matplotlib import pyplot
from matplotlib import ticker 
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler, MinMaxScaler, PowerTransformer,MaxAbsScaler
from scipy.stats import pearsonr

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import plot_cv_indices as vis

from xgboost.sklearn import XGBClassifier
le = LabelEncoder()

import warnings
warnings.filterwarnings('always')

class Music_Experiment:
    def __init__(self, data=None):
        self.raw_data = data #pandas dataframe
        self.groupings = None
        self.classes = None
        self.relevant_features = None
        self.processed_features = None

        self.orig_class = []
        self.pred_class = []

        self.scoring = scoring = {'Accuracy' : 'accuracy','Recall' : 'f1_weighted', 'Precision' : 'average_precision', 'ROC_AUC' : ' roc_auc'}
    
    def load_pandas_data(self, file_loc):
        try:
            self.raw_data = pd.read_pickle(file_loc)
            print("Data load successful")
        except:
            print("Data load failed, make sure it is a pandas pickled file")
        return True

    def split_data(self):
        #Get only low and mid level features
        df_rel_feat = self.raw_data.loc[:,'essentia_dissonance_mean':'mirtoolbox_roughness_pct_90']
        self.relevant_features = df_rel_feat.to_numpy()
        #feature set size
        print("Full feature size:",self.relevant_features.shape)

        self.groupings = (self.raw_data['pianist_id'].astype(str) + self.raw_data['segment_id'].astype(str)).to_numpy()
        #print(groupings.shape)

        self.classes = self.raw_data['quadrant'].to_numpy()
        return self.relevant_features, self.groupings, self.classes

    def preprocess_data(self, vt=False, scalar=True, thresh=.05):
        self.processed_features = self.relevant_features
        if vt:
            print("Old feature size:",self.relevant_features.shape)
            #remove features based on a variance threshold
            sel = VarianceThreshold(threshold=thresh)
            self.processed_features = sel.fit_transform(self.relevant_features)
            #size of thresholded feature set
            print("New VT feature size:",self.processed_features.shape)
        scaler = StandardScaler()
        self.processed_features = scaler.fit_transform(self.processed_features)
        return True

    def select_data(self, prepro=True):
        if prepro:
            self.preprocess_data()
            return self.processed_features, self.classes, self.groupings
        else:
            return self.relevant_features, self.classes, self.groupings
    
    def classification_report_with_accuracy_score(self,y_true, y_pred):
        self.orig_class.extend(y_true)
        self.pred_class.extend(y_pred)
        return accuracy_score(y_true, y_pred) # return accuracy score

    def get_model_list(self):
        models = list()
        models.append(DummyClassifier(strategy='most_frequent'))
        models.append(LogisticRegression(max_iter=1000))
        models.append(KNeighborsClassifier())
        models.append(DecisionTreeClassifier())
        models.append(SVC())
        models.append(GaussianNB())
        models.append(RandomForestClassifier())
        #models.append(GradientBoostingClassifier())
        models.append(XGBClassifier())
        return models

    def get_model_param_list(self):
        """
        model=KNeighborsClassifier()
        params={'n_neighbors': (1,5,8,10,20,30),
                'leaf_size': (1,5,8,10,20,30,40,50),
                'p': (1,2),
                'weights': ('uniform', 'distance'),
                'metric': ('minkowski', 'chebyshev')}
        """
        """
        model=SVC()
        params={'C': [0.1,1, 10, 100],
                'gamma': [1,0.1,0.01,0.001],
                'kernel': ['rbf', 'poly', 'sigmoid']}
        """

        """
        model=RandomForestClassifier()
        params={ 
                'n_estimators': [100, 200, 500],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [4,5,6,7,8],
                'criterion' :['gini', 'entropy']}
        """

        """
        model=XGBClassifier()
        params={
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [3, 4, 5]
                }
        """
        model=MLPClassifier()
        params = {
                'max_iter': [1000,1500,2000],
                'hidden_layer_sizes':[10,12,15], 
                'activation': ['tanh', 'relu'],
                'solver': ['lbfgs','sgd', 'adam'],
                'alpha': [0.0001, 0.05],
                'learning_rate': ['constant','adaptive'],
                }


        return model, params
    
    def get_best_models(self):
        models = list()
        models.append(MLPClassifier())
        models.append(KNeighborsClassifier(leaf_size=1, metric='minkowski',n_neighbors=5,p=1,weights='uniform'))
        models.append(SVC(C=1,gamma=0.001,kernel='rbf'))
        models.append(RandomForestClassifier(criterion='gini',max_depth=7,max_features='auto',n_estimators=500))
        models.append(XGBClassifier(colsample_bytree=1.0,gamma=1,max_depth=4,min_child_weight=5,subsample=1.0))
        return models

    def get_classifcation_report(self):
        report_dict = classification_report(self.orig_class,self.pred_class,output_dict=True)
        report_string = classification_report(self.orig_class,self.pred_class)
        conf_matr = confusion_matrix(self.orig_class,self.pred_class)

        return report_dict, report_string, conf_matr

    def evaluate_model(self,cv=GroupKFold(n_splits=5), to_test=None, prepro=False):

        X,y,groups = self.select_data(prepro)
        
        y = le.fit_transform(y) #label encoder to transoform for XGBoost error

        if to_test is not None: model = to_test
        else: model = GaussianNB()

        self.orig_class = []
        self.pred_class = []

        #cv_results = cross_val_score(model, X=X, y=y, cv=GroupKFold(n_splits=5), groups=groups)
        #cv_results = cross_validate(model, X=X, y=y, scoring=self.scoring, cv=GroupKFold(n_splits=5), groups=groups)
        cv_results = cross_val_score(model, X=X, y=y, cv=cv, groups=groups, scoring=make_scorer(self.classification_report_with_accuracy_score))
        
        report_dict, report_string, conf_matr = self.get_classifcation_report()

        return conf_matr, report_dict, report_string, np.mean(cv_results), cv_results.min(), cv_results.max()
    
    def evaluate_model_data(self,X,y,groups,cv=GroupKFold(n_splits=5)):
        
        y = le.fit_transform(y) #label encoder to transoform for XGBoost error

        model = XGBClassifier()

        self.orig_class = []
        self.pred_class = []

        cv_results = cross_val_score(model, X=X, y=y, cv=cv, groups=groups, scoring=make_scorer(self.classification_report_with_accuracy_score))
        
        report_dict, report_string, conf_matr = self.get_classifcation_report()

        return conf_matr, report_dict, report_string, np.mean(cv_results), cv_results.min(), cv_results.max()

    def eval_with_grid_search(self,cv=GroupKFold(n_splits=5), to_test=None, prepro=False, params=None):
        X,y,groups = self.select_data(prepro)
        
        y = le.fit_transform(y) #label encoder to transoform for XGBoost error

        if to_test is not None: model = to_test
        else: model = GaussianNB()

        self.orig_class = []
        self.pred_class = []

        cv = GridSearchCV(model, param_grid=params, cv=cv
            , scoring=make_scorer(self.classification_report_with_accuracy_score))
        
        result = cv.fit(X,y,groups=groups)
        
        report_dict, report_string, conf_matr = self.get_classifcation_report()

        return conf_matr, report_dict, report_string, result
    
    def test_folds(self):
        _,_,_,logo_mean,_,_ = self.evaluate_model(cv=LeaveOneGroupOut())
        print(f'Leave One Group Out mean: {logo_mean:.1%}')

        folds = range(2,26)
        # record mean and min/max of each set of results
        means, mins, maxs = list(),list(),list()
        means_, mins_, maxs_ = list(),list(),list()
        # evaluate each k value
        for k in folds:
            # define the test condition
            cv = GroupKFold(n_splits=k)
            cv2 = StratifiedGroupKFold(n_splits=k)
            # evaluate k value
            _,_,_,k_mean, k_min, k_max = self.evaluate_model(cv=cv)
            _,_,_,k_mean_, k_min_, k_max_ = self.evaluate_model(cv=cv2)
            # report performance
            print('>GroupKFold: folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
            print('>StratifiedGroupKFold: folds=%d, accuracy=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
            # store mean accuracy
            means.append(k_mean)
            means_.append(k_mean_)
            # store min and max relative to the mean
            mins.append(k_mean - k_min)
            maxs.append(k_max - k_mean)
            mins_.append(k_mean_ - k_min_)
            maxs_.append(k_max_ - k_mean_)

        fig, ax = pyplot.subplots()

        trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
        trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData

        # line plot of k mean values with min/max error bars
        ax.errorbar(folds, means, yerr=[mins, maxs], fmt='o', transform=trans1, label='GroupKFold')
        ax.errorbar(folds, means_, yerr=[mins_, maxs_], fmt='o', color='g', transform=trans2, label='StratifiedGroupKFold')
        # plot the ideal case in a separate color
        ax.plot(folds, [logo_mean for _ in range(len(folds))], color='r', label='LeaveOneGroupOut')
        pyplot.legend(loc='upper left')
        pyplot.title('Accuracy(mean,min,max) of Cross-Validation Methods')
        pyplot.xlabel('Number of Folds')
        pyplot.ylabel('Accuracy %')
        # show the plot
        pyplot.show()

    def visualize_folds(self):
        cmap_cv = pyplot.cm.coolwarm

        cvs = [GroupKFold, StratifiedGroupKFold]

        X, y, groups = self.select_data(False)
        groups = le.fit_transform(groups) 
        #groups = groups.astype(np.integer)

        for cv in cvs:
            fig, ax = pyplot.subplots(figsize=(6, 2))
            vis.plot_cv_indices(cv(5), X, y, groups, ax, 5)
            ax.legend(
                [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
                ["Testing set", "Training set"],
                loc=(1.02, 0.8),
            )
        # Make the legend fit
        pyplot.tight_layout()
        fig.subplots_adjust(right=0.7)
        pyplot.show()
        return
    
    def test_prepro(self, prepro_data=True):
        X, y, groups = self.select_data(prepro_data)

        y = le.fit_transform(y) #label encoder to transoform for XGBoost error

        #Not using kfolds/crossvalidation here
        gss=GroupShuffleSplit(n_splits=1, test_size=1/3)
        train_idx, test_idx = next(gss.split(X,y,groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(X_train.shape)
        print(X_test.shape)
        
        
        knn = KNeighborsClassifier().fit(X_train, y_train)
        print('---------KNN Default Setup--------')
        print('Training set score: ' + str(knn.score(X_train,y_train)))
        print('Test set score: ' + str(knn.score(X_test,y_test)))

        nb = GaussianNB().fit(X_train,y_train)
        print('---------Naive Bayes Default Setup--------')
        print('Training set score: ' + str(nb.score(X_train,y_train)))
        print('Test set score: ' + str(nb.score(X_test,y_test)))
        
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
        ])

        parameters = {'scaler': [StandardScaler(), MinMaxScaler(),
            Normalizer(), MaxAbsScaler()]
        }
        
        grid = GridSearchCV(pipe, parameters, cv=2).fit(X_train, y_train)
        print('---------KNN Default + Various Scalers--------')
        print('Training set score: ' + str(grid.score(X_train, y_train)))
        print('Test set score: ' + str(grid.score(X_test, y_test)))
        
        # Access the best set of parameters
        best_params = grid.best_params_
        print(best_params)
        # Stores the optimum model in best_pipe
        best_pipe = grid.best_estimator_
        print(best_pipe)
        
        result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
        print(result_df.columns)

        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
        ])

        parameters = {'scaler': [StandardScaler(), MinMaxScaler(),
            Normalizer(), MaxAbsScaler()]
        }
        
        grid = GridSearchCV(pipe, parameters, cv=2).fit(X_train, y_train)
        print('---------Naive Bayes Default + Various Scalers--------')
        print('Training set score: ' + str(grid.score(X_train, y_train)))
        print('Test set score: ' + str(grid.score(X_test, y_test)))
        
        # Access the best set of parameters
        best_params = grid.best_params_
        print(best_params)
        # Stores the optimum model in best_pipe
        best_pipe = grid.best_estimator_
        print(best_pipe)
        
        result_df = pd.DataFrame.from_dict(grid.cv_results_, orient='columns')
        print(result_df.columns)

        return 

    def test_models(self):
        #define our CVs
        logo = LeaveOneGroupOut()
        gkf = GroupKFold(n_splits=5)
        sgkf = StratifiedGroupKFold(n_splits=5)
        #Get the models to test
        models = self.get_model_list()
        #setup results lists
        logo_results, gkf_results, sgkf_results = list(),list()
        #loop through and evaluate models
        for model in models:
            print(f'>{type(model).__name__} evaluation starting...')
            _,_,_,logo_mean, logo_min, logo_max = self.evaluate_model(to_test=model, cv=logo)
            _,_,_,gkf_mean, gkf_min, gkf_max = self.evaluate_model(to_test=model, cv=gkf)
            _,_,_,sgkf_mean, sgkf_min, sgkf_max = self.evaluate_model(to_test=model, cv=sgkf)
            # check for invalid results
            if np.isnan(logo_mean) or np.isnan(gkf_mean) or np.isnan(sgkf_mean):
                continue
            logo_results.append(logo_mean)
            gkf_results.append(gkf_mean)
            sgkf_results.append(gkf_mean)

            print(f'Model: {type(model).__name__}: Leave One Group Out={logo_mean:.1%}, Group kFolds={gkf_mean:.1%}, Stratified Group kFolds={sgkf_mean:.1%}')

        # calculate the correlation between each test condition
        corr, _ = pearsonr(gkf_results, logo_results)
        print('Correlation: %.3f' % corr)
        # scatter plot of results
        pyplot.scatter(gkf_results, logo_results)
        # plot the line of best fit
        coeff, bias = np.polyfit(gkf_results, logo_results, 1)
        line = coeff * np.asarray(logo_results) + bias
        pyplot.plot(gkf_results, line, color='r')
        # label the plot
        pyplot.title('10-fold CV vs LOOCV Mean Accuracy')
        pyplot.xlabel('Mean Accuracy (Group 5-fold CV)')
        pyplot.ylabel('Mean Accuracy (LOGO)')
        # show the plot
        pyplot.show()
        return

    def test_models_main(self):
        #define our CV, based on experiments on splitting
        sgkf = StratifiedGroupKFold(n_splits=5)
        #Get the models to test
        models = self.get_model_list()
        #setup results lists
        model_names = list()
        sgkf_results, sgkf_report_dict, sgkf_report_string, = list(),list(),list()
        sgkf_means, sgkf_mins, sgkf_maxs = list(),list(),list()
        #loop through and evaluate models
        for model in models:
            print(f'...{type(model).__name__} evaluation starting...')
            conf_matr, report_dict, report_string, sgkf_mean, sgkf_min, sgkf_max = self.evaluate_model(to_test=model, cv=sgkf, prepro=True)
            # check for invalid results
            if np.isnan(sgkf_mean):
                continue
            # store mean accuracy
            sgkf_means.append(sgkf_mean)
            # store min and max relative to the mean
            sgkf_mins.append(sgkf_mean - sgkf_min)
            sgkf_maxs.append(sgkf_max - sgkf_mean)
            
            model_names.append(type(model).__name__)
            sgkf_results.append(conf_matr)
            sgkf_report_dict.append(report_dict)
            sgkf_report_string.append(report_string)

            print(f'> Model: {type(model).__name__}: Stratified Group kFolds={sgkf_mean:.1%} (Min:{sgkf_min:.1%},Max:{sgkf_max:.1%})')
            print('>')
            print(report_string)
            print(conf_matr)
            print('--------------------------------------------------------')

        fig, ax = pyplot.subplots()

        trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData

        # line plot of k mean values with min/max error bars
        ax.errorbar(model_names, sgkf_means, yerr=[sgkf_mins, sgkf_maxs], fmt='o', transform=trans1, 
                        label='Model Accuracy/Min/Max', capsize=2.0, markersize=10.0, markerfacecolor='orange')
        
        """
        # scatter plot of results
        pyplot.scatter(model_names, sgkf_means)
        """

        # label the plot
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        fig.autofmt_xdate(rotation=45)
        ax.set_ylim([0.20, 0.75])
        pyplot.grid(axis='y',linestyle='--')
        pyplot.title('Model Preformance summary')
        pyplot.xlabel('Model Tested')
        pyplot.ylabel('Mean Accuracy w/ StratifiedGroupKFold')
        # show the plot
        pyplot.show()
        return
    
    def param_tuning(self):
        
        #define our CV, based on experiments on splitting
        sgkf = StratifiedGroupKFold(n_splits=5)
        #Get the models to test
        model,p_grid = self.get_model_param_list()

        print(f'...{type(model).__name__} evaluation starting...')
        conf_matr, report_dict, report_string, result = self.eval_with_grid_search(to_test=model, cv=sgkf, prepro=True, params=p_grid)

        print(f'> Model: {type(model).__name__}:')
        print('>')
        #Parameter setting that gave the best results on the hold out data.
        print(result.best_params_ ) 
        #Mean cross-validated score of the best_estimator
        print('Best Score - Random Forest:', result.best_score_ )
        print(report_string)
        print(conf_matr)
        print('--------------------------------------------------------')

        return

    def of_analysis(self):
        
        #define our CV, based on experiments on splitting
        cv = StratifiedGroupKFold(n_splits=5)

        X,y,groups = self.select_data(True)
        
        y = le.fit_transform(y) #label encoder to transoform for XGBoost error

        # define the tree depths to evaluate
        values = [0.01,0.1,0.25,0.5,1,3,5,10,15]#[1,2,3,5,7,10,15] #[100,200,500,1000,1500,2000] #
        #[0.00001,0.0001,0.001,0.01,0.1,1]#[0.001,0.01,0.1,0.25,0.5,1,3,5] #[i for i in range(1, 25)]
        train_scores, test_scores = list(), list()
        # evaluate a decision tree for each depth
        for i in values:
            self.orig_class = []
            self.pred_class = []

            #cv_results = cross_validate(KNeighborsClassifier(n_neighbors=i), X=X, y=y, cv=cv, groups=groups, return_train_score=True)
            cv_results = cross_validate(XGBClassifier(gamma=i), X=X, y=y, cv=cv, groups=groups, return_train_score=True)
            test_scores.append(np.mean(cv_results['test_score']))
            train_scores.append(np.mean(cv_results['train_score']))
            
            print(f'> Model: {type(XGBClassifier()).__name__}: \n Test Score: {cv_results["test_score"]} \n Train Score: {cv_results["train_score"]}')
        # plot of train and test scores vs number of neighbors
        pyplot.plot(values, train_scores, '-o', label='Train')
        pyplot.plot(values, test_scores, '-o', label='Test')
        pyplot.title('Understanding Overfitting')
        pyplot.xlabel('Gamma')
        pyplot.ylabel('Accuracy')
        pyplot.legend()
        pyplot.show()
        return True

    def test_optimal_models(self):
        #define our CV, based on experiments on splitting
        sgkf = StratifiedGroupKFold(n_splits=5)
        #Get the models to test
        models = self.get_best_models()
        #setup results lists
        model_names = list()
        sgkf_results, sgkf_report_dict, sgkf_report_string, = list(),list(),list()
        sgkf_means, sgkf_mins, sgkf_maxs = list(),list(),list()
        #loop through and evaluate models
        for model in models:
            print(f'...{type(model).__name__} evaluation starting...')
            conf_matr, report_dict, report_string, sgkf_mean, sgkf_min, sgkf_max = self.evaluate_model(to_test=model, cv=sgkf, prepro=True)
            # check for invalid results
            if np.isnan(sgkf_mean):
                continue
            # store mean accuracy
            sgkf_means.append(sgkf_mean)
            # store min and max relative to the mean
            sgkf_mins.append(sgkf_mean - sgkf_min)
            sgkf_maxs.append(sgkf_max - sgkf_mean)
            
            model_names.append(type(model).__name__)
            sgkf_results.append(conf_matr)
            sgkf_report_dict.append(report_dict)
            sgkf_report_string.append(report_string)

            print(f'> Model: {type(model).__name__}: Stratified Group kFolds={sgkf_mean:.1%} (Min:{sgkf_min:.1%},Max:{sgkf_max:.1%})')
            print('>')
            print(report_string)
            print(conf_matr)
            print('--------------------------------------------------------')

        fig, ax = pyplot.subplots()

        trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData

        # line plot of k mean values with min/max error bars
        ax.errorbar(model_names, sgkf_means, yerr=[sgkf_mins, sgkf_maxs], fmt='o', transform=trans1, 
                        label='Model Accuracy/Min/Max', capsize=2.0, markersize=10.0, markerfacecolor='orange')

        # label the plot
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        fig.autofmt_xdate(rotation=45)
        ax.set_ylim([0.20, 0.75])
        pyplot.grid(axis='y',linestyle='--')
        pyplot.title('Model Preformance summary')
        pyplot.xlabel('Model Tested')
        pyplot.ylabel('Mean Accuracy w/ StratifiedGroupKFold')
        # show the plot
        pyplot.show()
        return

    def train_gem_gemmes(self):
        #Get only low and mid level features
        df_rel_feat = self.raw_data.loc[:,'essentia_dissonance_mean':'mirtoolbox_roughness_pct_90']
        X = df_rel_feat.to_numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        groups = (self.raw_data['pianist_id'].astype(str) + self.raw_data['segment_id'].astype(str)).to_numpy()

        y_list = self.raw_data.loc[:,'gems_wonder_binary':'gemmes_flow_binary'].to_numpy()
        class_list = ['gems_wonder_binary','gems_transcendence_binary','gems_tenderness_binary',
                        'gems_nostalgia_binary','gems_peacefulness_binary','gems_power_binary',
                        'gems_joyful_activation_binary','gems_tension_binary','gems_sadness_binary','gemmes_movement_binary',
                        'gemmes_force_binary','gemmes_interior_binary','gemmes_wandering_binary','gemmes_flow_binary']

        sgkf = StratifiedGroupKFold(n_splits=5)
        for i in range(0,14):
            conf_matr, report_dict, report_string, sgkf_mean, sgkf_min, sgkf_max = self.evaluate_model_data(X=X,y=y_list[:,i],groups=groups, cv=sgkf)

            print(f'> Model: {type(XGBClassifier()).__name__} + {class_list[i]} Accuracy:{sgkf_mean:.1%} (Min:{sgkf_min:.1%},Max:{sgkf_max:.1%})')
            print('>')
            #print(report_string)
            #print(conf_matr)
            #print('--------------------------------------------------------')


df_data = pd.read_pickle('Dataset/task_3_training_e8da4715deef7d56_f8b7378_pandas.pkl')

Exp = Music_Experiment(data=df_data)

Exp.split_data()

#results, report_dict, report_string, results_mean, results_max, results_min = Exp.evaluate_model()

#print(f"Accuracy: {report_dict['accuracy']:.1%}")
#print(results)
#print(classification_report(Exp.orig_class,Exp.pred_class))
#print(f"Accuracy: {np.mean(results):.1%} ({np.std(results):.1%})")


"""EXPERIMENTING FUNCTIONALITY"""
"""
print('-----------------WITHOUT VARIANCE THRESHOLD---------------------')
print('-----------------WITHOUT VARIANCE THRESHOLD---------------------')
Exp.test_prepro(prepro_data=False)
print('-----------------WITH VARIANCE THRESHOLD---------------------')
print('-----------------WITH VARIANCE THRESHOLD---------------------')
Exp.test_prepro(prepro_data=True)
"""
#Exp.test_folds() #5 or 8 folds
#Exp.visualize_folds()
#Exp.test_models()
#Exp.test_models_main()
Exp.param_tuning()
#Exp.of_analysis()
#Exp.test_optimal_models()
#Exp.train_gem_gemmes()


