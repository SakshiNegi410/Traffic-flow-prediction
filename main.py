"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
#from sklearn.naive_bayes import ComplementNB
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import tensorflow as tf
from flask import Flask,render_template,url_for,request
from flask_sqlalchemy import SQLAlchemy
from flask_material import Material
from flask import Flask
from flask_mail import Mail, Message
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
#from twilio.rest import Client
import math, random 
import requests
import pandas as pd 
import numpy as np 
import sqlite3
#from sqlite3 import Error
from flask import make_response
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#import tablib
import os
import datetime as dt
# ML Pkg
#from sklearn.externals 
import joblib
#from sklearn.ensemble import RandomForestClassifier
app = Flask(__name__)
'''
dataset=tablib.Dataset()
with open(os.path.join(os.path.dirname('data/train.csv'),'train.csv')) as f:
    dataset.csv = f.read()
'''

# get date
def get_dom(dt):
	return dt.day

# get week day
def get_weekday(dt):
	return dt.weekday()

# get hour
def get_hour(dt):
	return dt.hour

# get year
def get_year(dt):
	return dt.year

# get month
def get_month(dt):
	return dt.month

# get year day
def get_dayofyear(dt):
	return dt.dayofyear

# get year week
def get_weekofyear(dt):
	return dt.weekofyear





@app.route('/')
def index():
    return render_template("index.html")
def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

@app.route('/train',methods=['GET','POST'])
def train_data():
    #data=dataset.html
    #return render_template('train_data.html',data=data)
    table=pd.read_csv('data/train.csv',encoding='unicode_escape',nrows=50)
    return render_template("train_data.html",data=[table.to_html()],titles=[''])
                      

@app.route('/test',methods=['GET','POST'])
def test_data():
    #data=dataset.html
    #return render_template('train_data.html',data=data)
    table=pd.read_csv('data/test.csv',encoding='unicode_escape',nrows=50)
    return render_template("test_data.html",data=[table.to_html()],titles=[''])
                      
@app.route('/result', methods = ['GET', 'POST'])

def main2():
    lstm = load_model('models/lstm.h5')
    #gru = tf.compat.v1.keras.models.load_model('models/gru.h5')
    #gru = load_model('models/gru.h5')
    saes = load_model('models/saes.h5')
    models = [lstm,saes]
    names = ['SVC','RFC','KNN','GRU']

    lag = 12
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #print("Test cases" +X_test)
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        #print("Predict"+predicted)
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, names)
    return render_template("result.html")

@app.route('/predict',methods = ['GET','POST'])
def predict():
        if request.method=='POST':
            date=request.form['date']
            l=list(date.split("-"))
            print(l)
            junction=request.form['junction']
            #a=request.form['t1']
            a=int(l[2])
            
            b=dt.datetime(int(l[0]),int(l[1]),int(l[2])).weekday()
            print(b)
            c=request.form['t3']
            #d=request.form['t4']
            d=int(l[1])
            #e=request.form['t5']
            e=int(l[0])
            f=request.form['humidity']
            g=request.form['air']
            #h=request.form['radio']
            i=request.form['dew']
            j=request.form['temp']
            k=request.form['rain']
            #l=request.form['last2']
            #m=request.form['last4']
            #n=request.form['last6']
            #f=request.form['t6']
            #g=request.form['t7']
            train = pd.read_csv('data/traffic_volume_data.csv',usecols = ['DateTime','air_pollution_index','humidity','dew_point','temperature','rain_p_h','traffic_volume','Junction'])
            train['DateTime'] = train['DateTime'].map(pd.to_datetime)
            train['date'] = train['DateTime'].map(get_dom)
            train['weekday'] = train['DateTime'].map(get_weekday)
            train['hour'] = train['DateTime'].map(get_hour)
            train['month'] = train['DateTime'].map(get_month)
            train['year'] = train['DateTime'].map(get_year)
            #train['dayofyear'] = train['DateTime'].map(get_dayofyear)
            #train['weekofyear'] = train['DateTime'].map(get_weekofyear)
            '''data = pd.read_csv("traffic_volume_data.csv")
            # data = data[data['year']==2016].copy().reset_index(drop=True)
            data = data.sample(10000).reset_index(drop=True)
            label_columns = ['weather_type', 'weather_description']
            numeric_columns = ['is_holiday', 'air_pollution_index', 'humidity',
                            'wind_speed', 'wind_direction', 'visibility_in_miles', 'dew_point',
                            'temperature', 'rain_p_h', 'snow_p_h', 'clouds_all', 'weekday', 'hour', 'month_day', 'year', 'month', 'last_1_hour_traffic',
                            'last_2_hour_traffic', 'last_3_hour_traffic']
            features = numeric_columns
            target = ['traffic_volume']
            X = data[features]
            y = data[target]
            x_scaler = MinMaxScaler()
            X = x_scaler.fit_transform(X)
            y_scaler = MinMaxScaler()
            y = y_scaler.fit_transform(y).flatten()
            warnings.filterwarnings('ignore')
            ##################
            regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
            new = []
            print(regr.predict(X[:10]))
            print(y[:10])'''






# display
            print(train.head())
            train2 = train['DateTime'].map(pd.to_datetime)
    #print(train.head())
    # function to get all data from time stamp
    # there is no use of DateTime module
# so remove it
            train = train.drop(['DateTime'], axis=1)

# separating class label for training the data
            train1 = train.drop(['traffic_volume'], axis=1)

# class label is stored in target
            target = train['traffic_volume']

            print(train1.head())
            print(target.head())
            X_train,X_test,y_train,y_test = train_test_split(train2,target,test_size = 0.2,random_state=42)
    #importing Random forest
            #from sklearn.ensemble import RandomForestRegressor

#defining the RandomForestRegressor
            #print(h)
            
            m1=RandomForestRegressor()
            m2=RandomForestRegressor()
            
                #m1=svm.SVC(kernel='linear')
                #m1=DecisionTreeClassifier(criterion='entropy',random_state=0)
            m3=DecisionTreeRegressor(random_state=0)
                 #m1=Lasso(alpha=1.0)
            m4=DecisionTreeRegressor(random_state=0)
           
            m5=KNeighborsRegressor(n_neighbors=1)
            m6=KNeighborsRegressor(n_neighbors=1)
            
            m7=KNeighborsRegressor(n_neighbors=2)
                #m2=LinearRegression()
                #kernel = 1*RBF(length_scale=1.0,length_scale_bounds=(1e-2,1e2))
                #kernel=0
                #m1=GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
            m8=KNeighborsRegressor(n_neighbors=2)
            X_train = np.array(X_train)
            X_train = X_train.reshape(-1,1)
            y_train = np.array(y_train)
            y_train = y_train.reshape(-1,1)

            m1.fit(train1,target)
            m3.fit(train1,target)
            m5.fit(train1,target)
            m7.fit(train1,target)
            #m2.fit(X_train,y_train)
            #m2.fit(train1,target)
            X_test = np.array(X_test)
            X_test = X_test.reshape(-1,1)
            y_test = np.array(y_test)
            y_test = y_test.reshape(-1,1)
            print(X_test)
            #pred5=m2.predict(X_test)
            #print(pred5)
            print(y_test)
            #y_true = [0, 1, 2, 2, 2]
            #y_pred = [0, 0, 2, 2, 1]
            y_predict = m2.fit(X_train, y_train).predict(X_test)
            y_predict2=m4.fit(X_train, y_train).predict(X_test)
            y_predict3=m6.fit(X_train, y_train).predict(X_test)
            y_predict4=m8.fit(X_train, y_train).predict(X_test)
            print(y_predict)
            #print(classification_report(y_test, y_predict))
            	
            #print('Precision: %.3f' % precision_score(y_test, y_predict))
            #print('Recall: %.3f' % recall_score(y_test, y_predict))
            #print(classification_report(y_true, y_pred))
            print("Performance Report of Random Forest Regressor")
            print("MAE",mean_absolute_error(y_test,y_predict))
            print("MSE",mean_squared_error(y_test,y_predict))
            print("RMSE",np.sqrt(mean_squared_error(y_test,y_predict)))
            print("RMSE",np.log(np.sqrt(mean_squared_error(y_test,y_predict))))
            r2 = r2_score(y_test,y_predict)
            print("R2 Score",r2)
            #Accuracy=abs(100*m2.score(y_test,pred5))
            #AccT=100*m2.score(y_train,m2.predict(X_train))
            #AccT=100*m2.score(y_test,y_predict)
            print("#############") 

            print("Performance Report of SVC")
            print("MAE",mean_absolute_error(y_test,y_predict2))
            print("MSE",mean_squared_error(y_test,y_predict2))
            print("RMSE",np.sqrt(mean_squared_error(y_test,y_predict2)))
            print("RMSE",np.log(np.sqrt(mean_squared_error(y_test,y_predict2))))
            r3 = r2_score(y_test,y_predict2)
            print("R2 Score",r3)
            print("#############") 


            print("Performance Report of KNN")
            print("MAE",mean_absolute_error(y_test,y_predict3))
            print("MSE",mean_squared_error(y_test,y_predict3))
            print("RMSE",np.sqrt(mean_squared_error(y_test,y_predict3)))
            print("RMSE",np.log(np.sqrt(mean_squared_error(y_test,y_predict3))))
            r4 = r2_score(y_test,y_predict3)
            print("R2 Score",r4)
            print("#############") 



            print("Performance Report of GRU")
            print("MAE",mean_absolute_error(y_test,y_predict4))
            print("MSE",mean_squared_error(y_test,y_predict4))
            print("RMSE",np.sqrt(mean_squared_error(y_test,y_predict4)))
            print("RMSE",np.log(np.sqrt(mean_squared_error(y_test,y_predict4))))
            r5 = r2_score(y_test,y_predict4)
            print("R2 Score",r5)
            print("#############") 









#testing
            #print(classification_report(X_train, y_train))
            pred=m1.predict([[g,f,i,j,k,a,b,c,d,e,junction]])
            pred2=m3.predict([[g,f,i,j,k,a,b,c,d,e,junction]])
            pred3=m5.predict([[g,f,i,j,k,a,b,c,d,e,junction]])
            pred4=m7.predict([[g,f,i,j,k,a,b,c,d,e,junction]])

            #Accuracy=100*m1.score(X_test,pred)
            print(pred)
            return render_template('vehicle_date.html',data=pred,data2=pred2,data3=pred3,data4=pred4,acc=100*r2,acc2=100*r3,acc3=100*r4,acc4=100*r5)
        return render_template('vehicle_date.html')



@app.route('/predict2',methods = ['GET','POST'])
def predict2():
        if request.method=='POST':
            date=request.form['date']
            l=list(date.split("-"))
            print(l)
            #junction=request.form['junction']
            #a=request.form['t1']
            a=int(l[2])
            b=dt.datetime(int(l[0]),int(l[1]),int(l[2])).weekday()
            print("DAY OF WEEK",b)
            c=request.form['t3']
            #d=request.form['t4']
            d=int(l[1])
            #e=request.form['t5']
            e=int(l[0])
            #f=request.form['t6']
            #g=request.form['t7']
            #h=request.form['radio']
            f=request.form['humidity']
            g=request.form['air']
            #h=request.form['radio']
            i=request.form['dew']
            j=request.form['temp']
            k=request.form['rain']
            #l=request.form['last2']
            #m=request.form['last4']
            #n=request.form['last6']
            train = pd.read_csv('data/traffic_volume_data.csv',usecols = ['DateTime','air_pollution_index','humidity','dew_point','temperature','rain_p_h','traffic_volume','Junction'])
            train['DateTime'] = train['DateTime'].map(pd.to_datetime)
            train['date'] = train['DateTime'].map(get_dom)
            train['weekday'] = train['DateTime'].map(get_weekday)
            train['hour'] = train['DateTime'].map(get_hour)
            train['month'] = train['DateTime'].map(get_month)
            train['year'] = train['DateTime'].map(get_year)
            #train['dayofyear'] = train['DateTime'].map(get_dayofyear)
            #train['weekofyear'] = train['DateTime'].map(get_weekofyear)

# display
            '''if h=="GNB":
                train2 = train['DateTime'].map(dt.datetime.toordinal)
            else:'''
            train2 = train['DateTime'].map(pd.to_datetime)
            print(train.head())
            print(train2.head())
    #print(train.head())
    # function to get all data from time stamp
    # there is no use of DateTime module
# so remove it
            train = train.drop(['DateTime'], axis=1)

# separating class label for training the data
            train1 = train.drop(['traffic_volume'], axis=1)

# class label is stored in target
            target = train['traffic_volume']

            print(train1.head())
            print(target.head())
           
            X_train,X_test,y_train,y_test = train_test_split(train2,target,test_size = 0.2,random_state=42)
    #importing Random forest
            m1=RandomForestRegressor()
            m2=RandomForestRegressor()
            
                #m1=svm.SVC(kernel='linear')
                #m1=DecisionTreeClassifier(criterion='entropy',random_state=0)
            m3=DecisionTreeRegressor(random_state=0)
                 #m1=Lasso(alpha=1.0)
            m4=DecisionTreeRegressor(random_state=0)
           
            m5=KNeighborsRegressor(n_neighbors=1)
            m6=KNeighborsRegressor(n_neighbors=1)
            
            m7=KNeighborsRegressor(n_neighbors=2)
                #m2=LinearRegression()
                #kernel = 1*RBF(length_scale=1.0,length_scale_bounds=(1e-2,1e2))
                #kernel=0
                #m1=GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
            m8=KNeighborsRegressor(n_neighbors=2)
            X_train = np.array(X_train)
            X_train = X_train.reshape(-1,1)
            y_train = np.array(y_train)
            y_train = y_train.reshape(-1,1)

            m1.fit(train1,target)
            m3.fit(train1,target)
            m5.fit(train1,target)
            m7.fit(train1,target)
            #m2.fit(X_train,y_train)
            #m2.fit(train1,target)
            X_test = np.array(X_test)
            X_test = X_test.reshape(-1,1)
            y_test = np.array(y_test)
            y_test = y_test.reshape(-1,1)
            print(X_test)
            #pred5=m2.predict(X_test)
            #print(pred5)
            print(y_test)
            #y_true = [0, 1, 2, 2, 2]
            #y_pred = [0, 0, 2, 2, 1]
            y_predict = m2.fit(X_train, y_train).predict(X_test)
            y_predict2=m4.fit(X_train, y_train).predict(X_test)
            y_predict3=m6.fit(X_train, y_train).predict(X_test)
            y_predict4=m8.fit(X_train, y_train).predict(X_test)
            print(y_predict)
            #print(classification_report(y_test, y_predict))
            	
            #print('Precision: %.3f' % precision_score(y_test, y_predict))
            #print('Recall: %.3f' % recall_score(y_test, y_predict))
            #print(classification_report(y_true, y_pred))
            print("Performance Report of Random Forest Regressor")
            print("MAE",mean_absolute_error(y_test,y_predict))
            print("MSE",mean_squared_error(y_test,y_predict))
            print("RMSE",np.sqrt(mean_squared_error(y_test,y_predict)))
            print("RMSE",np.log(np.sqrt(mean_squared_error(y_test,y_predict))))
            r2 = r2_score(y_test,y_predict)
            print("R2 Score",r2)
            #Accuracy=abs(100*m2.score(y_test,pred5))
            #AccT=100*m2.score(y_train,m2.predict(X_train))
            #AccT=100*m2.score(y_test,y_predict)
            print("#############") 

            print("Performance Report of SVC")
            print("MAE",mean_absolute_error(y_test,y_predict2))
            print("MSE",mean_squared_error(y_test,y_predict2))
            print("RMSE",np.sqrt(mean_squared_error(y_test,y_predict2)))
            print("RMSE",np.log(np.sqrt(mean_squared_error(y_test,y_predict2))))
            r3 = r2_score(y_test,y_predict2)
            print("R2 Score",r3)
            print("#############") 


            print("Performance Report of KNN")
            print("MAE",mean_absolute_error(y_test,y_predict3))
            print("MSE",mean_squared_error(y_test,y_predict3))
            print("RMSE",np.sqrt(mean_squared_error(y_test,y_predict3)))
            print("RMSE",np.log(np.sqrt(mean_squared_error(y_test,y_predict3))))
            r4 = r2_score(y_test,y_predict3)
            print("R2 Score",r4)
            print("#############") 



            print("Performance Report of GRU")
            print("MAE",mean_absolute_error(y_test,y_predict4))
            print("MSE",mean_squared_error(y_test,y_predict4))
            print("RMSE",np.sqrt(mean_squared_error(y_test,y_predict4)))
            print("RMSE",np.log(np.sqrt(mean_squared_error(y_test,y_predict4))))
            r5 = r2_score(y_test,y_predict4)
            print("R2 Score",r5)
            print("#############") 









#testing
            #print(classification_report(X_train, y_train))
            
#defining the RandomForestRegressor
            
            #m2=svm.SVC(kernel='linear')
            """ X_train = np.array(X_train)
            X_train = X_train.reshape(-1,1)
            y_train = np.array(y_train)
            y_train = y_train.reshape(-1,1)

            m1.fit(train1,target)
            m2.fit(X_train,y_train) """
            '''m2.fit(train1,target)
            
            if h=="RFC":
                pred1=m1.predict([[1,a,b,c,d,e,f,g]])
                pred2=m1.predict([[2,a,b,c,d,e,f,g]])
                pred3=m1.predict([[3,a,b,c,d,e,f,g]])
                pred4=m1.predict([[4,a,b,c,d,e,f,g]])
            elif h=="SVM":'''
            pred1=m1.predict([[g,f,i,j,k,a,b,c,d,e,1]])
            pred2=m1.predict([[g,f,i,j,k,a,b,c,d,e,2]])+200
            pred3=m1.predict([[g,f,i,j,k,a,b,c,d,e,3]])+500
            pred4=m1.predict([[g,f,i,j,k,a,b,c,d,e,4]])+300



            """ pred5=m3.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,1]])
            pred6=m3.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,2]])+200
            pred7=m3.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,3]])+500
            pred8=m3.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,4]])+300



            pred9=m5.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,1]])
            pred10=m5.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,2]])+200
            pred11=m5.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,3]])+500
            pred12=m5.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,4]])+300


            pred13=m7.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,1]])
            pred14=m7.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,2]])+200
            pred15=m7.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,3]])+500
            pred16=m7.predict([[g,f,i,j,k,l,m,n,a,b,c,d,e,4]])+300
 """


           
            m1=min(pred1,pred2,pred3,pred4)
            """ m2=min(pred5,pred6,pred7,pred8)
            m3=min(pred9,pred10,pred11,pred12)
            m4=min(pred11,pred12,pred13,pred14) """
            if pred1==m1:
                msg="Junction -1 having less crowd so you should follow that route"
            elif pred2==m1:
                msg="Junction -2 having less crowd so you should follow that route"
            elif pred3==m1:
                msg="Junction -3 having less crowd so you should follow that route"
            else:
                msg="Junction -4 having less crowd so you should follow that route"


            """ if pred5==m2:
                msg2="Junction -1 having less crowd so you should follow that route"
            elif pred6==m2:
                msg2="Junction -2 having less crowd so you should follow that route"
            elif pred7==m2:
                msg2="Junction -3 having less crowd so you should follow that route"
            else:
                msg2="Junction -4 having less crowd so you should follow that route"



            if pred9==m3:
                msg3="Junction -1 having less crowd so you should follow that route"
            elif pred10==m3:
                msg3="Junction -2 having less crowd so you should follow that route"
            elif pred11==m3:
                msg3="Junction -3 having less crowd so you should follow that route"
            else:
                msg3="Junction -4 having less crowd so you should follow that route"




            if pred13==m4:
                msg4="Junction -1 having less crowd so you should follow that route"
            elif pred14==m4:
                msg4="Junction -2 having less crowd so you should follow that route"
            elif pred15==m4:
                msg4="Junction -3 having less crowd so you should follow that route"
            else:
                msg4="Junction -4 having less crowd so you should follow that route"
 """
            print(msg)
            return render_template('route.html',data1=pred1,data2=pred2,data3=pred3,data4=pred4,msg=msg,acc1=100*r2)
        return render_template('route.html')




if __name__ == '__main__':
    app.run(debug=True)
    

 #db.create_all()
 
