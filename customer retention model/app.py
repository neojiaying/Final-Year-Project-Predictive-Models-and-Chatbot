# pip install pipwin 
# pipwin install xgBoost

# pip install joblib

# pip install imblearn

from flask import Flask,request,send_file,render_template,url_for,abort
from werkzeug.utils import secure_filename

import pandas as pd
import joblib
import datetime as dt

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier

app = Flask(__name__)


@app.route('/')
def home():
 return render_template('home.html')

@app.route('/test', methods =['POST','GET'])
def getMetrics():
    if request.method == "POST":
        try:
            # Timer
            start = dt.datetime.now()
            # Import files
            train = pd.read_csv(request.files['train_file'])
            # Check for empty train file
            X = train.drop('labels', 1)
            y = train['labels']
            # Split into Train and Test dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            # SMOTE
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)
            # Train models
            # XGBoost
            clf_xgb = XGBClassifier()
            clf_xgb.fit(X_train, y_train)
            accuracy_xgb = clf_xgb.score(X_test, y_test)
            metrics_xgb = precision_recall_fscore_support(y_test, clf_xgb.predict(X_test), average='macro')
            joblib.dump(clf_xgb, 'xgb.pkl')
            # Random Forest
            clf_rf = RandomForestClassifier()
            clf_rf.fit(X_train,y_train)
            accuracy_rf = clf_rf.score(X_test, y_test)
            metrics_rf = precision_recall_fscore_support(y_test, clf_rf.predict(X_test), average='macro')
            joblib.dump(clf_rf, 'rf.pkl')
            # Extra Trees
            clf_et = ExtraTreesClassifier()
            clf_et.fit(X_train,y_train)
            accuracy_et = clf_et.score(X_test, y_test)
            metrics_et = precision_recall_fscore_support(y_test, clf_et.predict(X_test), average='macro')
            joblib.dump(clf_et, 'et.pkl')
            # End timer
            time_taken = str(dt.datetime.now()-start)

            if accuracy_xgb > accuracy_rf:
                highest_accuracy_model = "XGBoost"
            elif accuracy_et > accuracy_xgb:
                highest_accuracy_model = "Extra Trees"
            else:
                highest_accuracy_model = "Random Forest"

            if metrics_xgb[2] > metrics_rf[2]:
                highest_fscore_model = "XGBoost"
            elif metrics_et[2] > metrics_xgb[2]:
                highest_fscore_model = "Extra Trees"
            else:
                highest_fscore_model = "Random Forest"
            return render_template("metrics.html", metrics_xgb = metrics_xgb, accuracy_xgb = accuracy_xgb, metrics_rf = metrics_rf, accuracy_rf = accuracy_rf,metrics_et=metrics_et,accuracy_et=accuracy_et,time_taken=time_taken,highest_accuracy_model=highest_accuracy_model,highest_fscore_model=highest_fscore_model) 
        except:
            abort(400)

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

@app.route('/exportET', methods =['POST','GET'])
def predictET():
    # Load Model
    clf_et = joblib.load('et.pkl')
    if request.method == 'POST':
        try:
            test = pd.read_csv(request.files['test_file'])
            pred = clf_et.predict(test)
            test['predictions'] = pred
            test.to_csv('test_predictions_et.csv')
            path = "test_predictions_et.csv"
            return send_file(path, as_attachment = True)
        except:
            abort(400)

@app.errorhandler(400)
def page_not_found(e):
    # note that we set the 400 status explicitly
    return render_template('400.html'), 400

if __name__ == '__main__':
    app.run(debug=True)