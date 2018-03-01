import numpy as np
import os
import cPickle as pickle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import xgboost as xgb

WORK_DIR = "/home/ubuntu/Documents/Model_Build/"

def balance_data(X_train, y_train, X_valid, y_valid, X_test, y_test):
    X_train_pos = X_train[y_train==1]
    X_train = np.vstack([X_train_pos, X_train[y_train==0][:len(X_train_pos)]])
    y_train_pos = y_train[y_train==1]
    y_train = np.hstack([y_train_pos, y_train[y_train==0][:len(y_train_pos)]])

    X_valid_pos = X_valid[y_valid==1]
    X_valid = np.vstack([X_valid_pos, X_valid[y_valid==0][:len(X_valid_pos)]])
    y_valid_pos = y_valid[y_valid==1]
    y_valid = np.hstack([y_valid_pos, y_valid[y_valid==0][:len(y_valid_pos)]])

    X_test_pos = X_test[y_test==1]
    X_test = np.vstack([X_test_pos, X_test[y_test==0][:len(X_test_pos)]])
    y_test_pos = y_test[y_test==1]
    y_test = np.hstack([y_test_pos, y_test[y_test==0][:len(y_test_pos)]])
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def build_train_test_data(df, target_col, balanced=False, train_valid_split=0.3):
    train_ids = df[df['Train_Flag']]['Polygon_id'].unique()
    train_ids, valid_ids = train_test_split(train_ids, test_size = train_valid_split)
    X_train = df[df['Polygon_id'].isin(train_ids)].drop(['Polygon_id','Local_Filepath', 'Train_Flag', target_col], axis=1)
    X_train = np.apply_along_axis(lambda x: np.hstack(x), 1, X_train.values)
    y_train = df[df['Polygon_id'].isin(train_ids)].loc[df['Train_Flag'], target_col].astype('float').values

    X_valid = df[df['Polygon_id'].isin(valid_ids)].drop(['Polygon_id','Local_Filepath', 'Train_Flag', target_col], axis=1)
    X_valid = np.apply_along_axis(lambda x: np.hstack(x), 1, X_valid.values)
    y_valid = df[df['Polygon_id'].isin(valid_ids)].loc[df['Train_Flag'], target_col].astype('float').values

    test = df[~df['Train_Flag']].drop(['Polygon_id','Local_Filepath', 'Train_Flag', target_col], axis=1)
    X_test = np.apply_along_axis(lambda x: np.hstack(x), 1, test.values)
    #X_test = np.array(df[~df['Train_Flag']].drop(['Local_Filepath', 'Train_Flag', target_col], axis=1).values)
    y_test = df.loc[~df['Train_Flag'], target_col].astype('float').values
    if balanced:
        X_train, y_train, X_valid, y_valid, X_test, y_test = balance_data(X_train, y_train, X_valid, y_valid, X_test, y_test)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def benchmark_model(X_train, y_train, X_test, y_test):
    lr = LogisticRegression()
    rf = RandomForestClassifier(max_depth=3)
    xgb_tree = xgb.XGBClassifier(max_depth=3, objective='binary:logistic')    
    models = [('lr', lr), ('rf', rf), ('xgb', xgb_tree)]
    eclf = VotingClassifier(estimators=models, voting='hard')
    for model in models + [('eclf',eclf)]:
        model[1].fit(X_train, y_train)        
        print "*----------*"
        print "{}: train error: {}".format(model[0],model[1].score(X_train, y_train))
        print "{}: test error: {}".format(model[0],model[1].score(X_test, y_test))

    print "*----------*"
    print "5-fold Cross-Validation"
    for clf, label in zip([lr, rf, xgb_tree, eclf], ['Logistic Regression', 'Random Forest', 'XGBoost', 'Ensemble']):        
        scores = cross_val_score(clf, np.vstack([X_train,X_test]), np.hstack([y_train, y_test]), cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    return eclf

def build_and_score_model(feature_df, model_function, target_column):
    X_train, y_train, X_test, y_test, _, _ = build_train_test_data(feature_df, target_column, train_valid_split=0.3, balanced=True)
    imputer = Imputer(strategy='mean',axis=0)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    model = model_function(X_train, y_train, X_test, y_test)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print "*------ Training Confusion Matrix -------*"
    print confusion_matrix(y_train, y_train_pred)
    print "*------ Testing Confusion Matrix  ------*"
    print confusion_matrix(y_test, y_test_pred)
    return model

if __name__ == '__main__':
    experiment_name = 'lbp_entr_dop10.pkl'
    with open(os.path.join(WORK_DIR, 'Experiments', experiment_name), 'r') as fp:
        (feature_df, features) = pickle.load(fp)
    target_column = 'Young_Tree_Flag'    
    
    model = build_and_score_model(feature_df, benchmark_model, target_column)
