from flask import Flask,request,send_file,render_template,redirect,session,abort, flash

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import datetime as dt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support

app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = 'super secret key'
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/feature')
def feature():
    return render_template('feature.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/features_importance', methods =['POST','GET'])
def cut_features():
    if request.method == 'POST':
        try:
            start = dt.datetime.now()
            #Import files
            df_train = pd.read_csv(request.files['train_file'])
            df_train.to_csv("train.csv")
            X_train = df_train.drop('labels', 1)
            Y_train = df_train['labels']
            model = LogisticRegression(penalty='l1', solver='saga', C=2, multi_class='multinomial', n_jobs=-1, random_state=42)

            model.fit(X_train, Y_train)
            importance_values = np.absolute(model.coef_)
            importance_features_sorted = pd.DataFrame(importance_values.reshape([-1, len(X_train.columns)]), columns=X_train.columns).mean(axis=0).sort_values(ascending=False).to_frame()
            importance_features_sorted.rename(columns={0: 'feature_importance'}, inplace=True)
            importance_features_sorted['ranking']= importance_features_sorted['feature_importance'].rank(ascending=False)
            # End timer
            time_taken = str(dt.datetime.now()-start)

            plt.title('Feature importance ranked by number of features by model')
            sns.lineplot(data=importance_features_sorted, x='ranking', y='feature_importance')
            plt.xlabel("Number of features selected")
            plt.savefig('./static/features_importance.png')

            importance_features_sorted.to_csv("feature_ranking.csv")
            return redirect("/feature")
        except:
            abort(400)
    


@app.route('/features', methods =['POST','GET'])
def get_features():
    df_train = pd.read_csv("train.csv")
    importance_features_sorted = pd.read_csv("feature_ranking.csv")
    importance_features_sorted = importance_features_sorted.rename(columns={"Unnamed: 0":"features"})
    if request.method == 'POST':
        if request.form['important_features'].isnumeric():
            number_of_features = int(request.form['important_features'])
            X = df_train.drop('labels', 1)
            target = df_train['labels']

            estimator = LogisticRegression(penalty='l1', solver='saga', C=2, multi_class='multinomial', n_jobs=-1, random_state=42)
            rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedShuffleSplit(1, test_size=.2,random_state=42), scoring='accuracy')
            select_features_by_model = importance_features_sorted[importance_features_sorted['ranking']<=number_of_features]['features'].tolist()
            rfecv.fit(X[select_features_by_model], target)
            #print('Optimal number of features: {}'.format(rfecv.n_features_))
            
            plt.figure(figsize=(16, 9))
            plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
            plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
            plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
            plt.savefig('./static/features.png')

            rfecv_df = pd.DataFrame({'col': select_features_by_model})
            rfecv_df['rank'] = np.nan
            for index, support in enumerate(rfecv.get_support(indices=True)):
                rfecv_df.loc[support, 'rank'] = index
            for index, rank in enumerate(rfecv.ranking_ -2):
                if rank >= 0:
                    rfecv_df.loc[index, 'rank'] = rfecv.n_features_ + rank
            rfecv_df.to_csv('features.csv')
            return redirect("/model")
        else:
            flash("Please enter a digit for the number of features to select!")
            return redirect("/feature")
   


@app.route('/results', methods =['POST','GET'])
def predict():
    if request.method == "POST":
        if request.form['features'].isnumeric():
            start = dt.datetime.now()
            train = pd.read_csv("train.csv")
            X = train.drop('labels', 1)
            y = train['labels']
            #features
            num_features = int(request.form['features'])
            features_name_to_keep = pd.read_csv("features.csv")
            features_name_to_keep = features_name_to_keep[features_name_to_keep['rank']<num_features]["col"].to_list()
            #only keep selected features from feature selection
            X = X[features_name_to_keep]

            # Split into Train and Test dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # SMOTE
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)
            #model training
            #xgboost
            clf_xgb = XGBClassifier()
            clf_xgb.fit(X_train, y_train)
            accuracy_xgb = clf_xgb.score(X_test, y_test)
            metrics_xgb = precision_recall_fscore_support(y_test, clf_xgb.predict(X_test), average='macro')
            joblib.dump(clf_xgb, 'xgb.pkl')
            # Random Forest
            clf_rf = RandomForestClassifier(n_estimators= 400,
                                            min_samples_split= 2,
                                            min_samples_leaf= 1,
                                            max_features= 'sqrt',
                                            max_depth= None,
                                            bootstrap= False)
            clf_rf.fit(X_train,y_train)
            accuracy_rf = clf_rf.score(X_test, y_test)
            metrics_rf = precision_recall_fscore_support(y_test, clf_rf.predict(X_test), average='macro')
            joblib.dump(clf_rf, 'rf.pkl')
            # Adaboost
            clf_ab = AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth=2),
                    n_estimators=200
                )
            clf_ab.fit(X_train,y_train)
            accuracy_ab = clf_ab.score(X_test, y_test)
            metrics_ab = precision_recall_fscore_support(y_test, clf_ab.predict(X_test), average='macro')
            joblib.dump(clf_ab, 'ab.pkl')
            # End timer
            time_taken = str(dt.datetime.now()-start)

            if accuracy_xgb > accuracy_rf:
                highest_accuracy_model = "XGBoost"
            elif accuracy_ab > accuracy_xgb:
                highest_accuracy_model = "AdaBoost"
            else:
                highest_accuracy_model = "Random Forest"

            if metrics_xgb[2] > metrics_rf[2]:
                highest_fscore_model = "XGBoost"
            elif metrics_ab[2] > metrics_xgb[2]:
                highest_fscore_model = "AdaBoost"
            else:
                highest_fscore_model = "Random Forest"
            return render_template("metrics.html", metrics_xgb = metrics_xgb, accuracy_xgb = accuracy_xgb, metrics_rf = metrics_rf, accuracy_rf = accuracy_rf,metrics_ab=metrics_ab,accuracy_ab=accuracy_ab,time_taken=time_taken,highest_accuracy_model=highest_accuracy_model,highest_fscore_model=highest_fscore_model) 
        else:
            flash("Please enter a digit for the number of features to use!")
            return redirect("/model")


@app.route('/exportXGB', methods =['POST','GET'])
def predictXGB():
    # Load Model
    clf_xgb = joblib.load('xgb.pkl')
    if request.method == 'POST':
        try:
            test = pd.read_csv(request.files['test_file'])
            pred = clf_xgb.predict(test)
            test['predictions'] = pred
            test.to_csv('test_predictions_xgb.csv')
            path = "test_predictions_xgb.csv"
            return send_file(path, as_attachment = True)
        except:
            abort(400)

@app.route('/exportRF', methods =['POST','GET'])
def predictRF():
    # Load Model
    clf_rf = joblib.load('rf.pkl')
    if request.method == 'POST':
        try:
            test = pd.read_csv(request.files['test_file'])
            pred = clf_rf.predict(test)
            test['predictions'] = pred
            test.to_csv('test_predictions_rf.csv')
            path = "test_predictions_rf.csv"
            return send_file(path, as_attachment = True)
        except:
            abort(400)

@app.route('/exportAB', methods =['POST','GET'])
def predictET():
    # Load Model
    clf_et = joblib.load('ab.pkl')
    if request.method == 'POST':
        try:
            test = pd.read_csv(request.files['test_file'])
            pred = clf_et.predict(test)
            test['predictions'] = pred
            test.to_csv('test_predictions_ab.csv')
            path = "test_predictions_ab.csv"
            return send_file(path, as_attachment = True)
        except:
            abort(400)

@app.errorhandler(400)
def page_not_found(e):
    # note that we set the 400 status explicitly
    return render_template('400.html'), 400

if __name__ == '__main__':
    app.run(debug=True)
