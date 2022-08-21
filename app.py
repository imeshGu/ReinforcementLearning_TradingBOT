from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from requests import head
from wtforms import FileField,SubmitField 
from werkzeug.utils import secure_filename
import os 
from wtforms.validators import InputRequired

from stable_baselines3 import A2C

import numpy as np
import pandas as pd

from gym_anytrading.envs import StocksEnv#customer overlay(customiz env to able add indicators)
from finta import TA#technical indicators

import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

app = Flask(__name__)
app.config['SECRET_KEY'] = "secret"
app.config['UPLOAD_FOLDER'] = 'static\files'

#add custom singals to df
def add_signals(env):
    start = env.frame_bound[0] - env.window_size #grabing the first inex
    end = env.frame_bound[1]#grabing the ending indexes
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]#prices from start to end
    signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV']].to_numpy()[start:end]#
    return prices, signal_features

#form
class UploadCSVForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

#custom env
class MyCustomEnv(StocksEnv):
    _process_data = add_signals #native function in stock trading env
    #over write it with add_signals


def plot_png(env):
    plt.figure(figsize=(15,6))
    plt.cla()
    env.render_all()
    plt.show()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')




@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadCSVForm()
    dictInfo = {}
    if form.validate_on_submit():
        file = form.file.data#grab file
        print(type(file))
        #save the file 
        name = secure_filename(file.filename)
        print(type(name))
        file.save(name)
        print("file is saved")
        print(name)
        
        #saving as df in memory
        #df = pd.read_excel(name)
        df = pd.read_csv(name)
        print("file is now dataframe")
        print(df.head())
        #preprocessing
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', ascending=True, inplace=True)
        df.set_index('Date', inplace=True)
        df['Volume'] = df['Volume'].apply(lambda x: float(x.replace(",", "")))
        #adding custom indicators
        df['SMA'] = TA.SMA(df, 12)#12- no of period we want from simple moving avg
        df['RSI'] = TA.RSI(df)
        df['OBV'] = TA.OBV(df)
        df.fillna(0, inplace=True) 
        print("data is preprocessed")

        model = A2C.load("model")
        print("model loaded")

        env = MyCustomEnv(df=df, window_size=12, frame_bound=(80,120))
        obs = env.reset()
        print("env is reset")
        while True: 
            obs = obs[np.newaxis, ...]
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                print("info", info)
                dictInfo = info

                #plt.figure(figsize=(15,6))
                #plt.cla() 
                #env.render_all()
                #plt.savefig('templates/books_read.jpg')
                #graph = plt.show(block=False)
                #plt.close()


                break

                
            
                
        #file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))        
        return render_template("result.html",dictInfo=dictInfo)
    return render_template('form.html', form=form)
   
if __name__ == "__main__":
    app.run(debug=True)#errors will pop up in page