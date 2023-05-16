import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.metrics import roc_auc_score

from imblearn.under_sampling import RandomUnderSampler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import AUC
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup

st.title('Pitching Fatigue Predictor')
last = st.text_input('Pitcher Name: Last')
first = st.text_input('Pitcher Name: First')

def load_pitcher(last,first):
    pitcher = statcast_pitcher('2015-01-01','2023-05-09',
                               playerid_lookup(str(last),str(first))['key_mlbam'][0])
    pitcher['game_date'] = pd.to_datetime(pitcher['game_date'])
    pitcher['pitch_no'] = pitcher.groupby('game_date').cumcount() + 1
    pitcher['swing'] = pitcher['description'].apply(lambda x: 1 if x == 'swinging_strike' else 0)
    pitcher['ball'] = pitcher['type'].apply(lambda x: 1 if x == 'B' else 0)
    pitcher = pitcher[['pitch_type','release_speed','pfx_x','pfx_z',
            'plate_x','plate_z','vx0','vy0','vz0',
            'ax','ay','az','sz_top','sz_bot',
            'release_spin_rate','release_extension','effective_speed','inning',
        'release_pos_x','release_pos_y','release_pos_z','pitch_no','swing','ball']]
    pitcher.dropna(inplace = True)
    pitches = []
    for i in pitcher['pitch_type'].unique():
        if pitcher['pitch_type'].value_counts(normalize = True).get(key = i) > 0.05:
            pitches.append(i)
    return pitcher, pitches

def model_ball(last,first):
    models = {}
    pitcher, pitches = load_pitcher(last,first)
    for pitch in pitches:
        X = pitcher[pitcher['pitch_type'] == pitch]
        y = X['ball']

        # final position of ball and size of strike zone gives away the result
        X = X.drop(columns = ['swing','ball','pitch_type','plate_x','plate_z','sz_top','sz_bot'])
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42)
        pipe = Pipeline(steps = [
            ('ss',StandardScaler()),
            ('poly',PolynomialFeatures())
        ])

        X_train = pipe.fit_transform(X_train)
        X_test = pipe.transform(X_test)

        X_train = pd.DataFrame(X_train, columns = pipe.get_feature_names_out())
        X_test = pd.DataFrame(X_test, columns = pipe.get_feature_names_out())

        X_train, y_train = RandomUnderSampler().fit_resample(X_train, y_train)

        model = Sequential()
        model.add(Dense(32,activation = 'relu'))
        model.add(Dense(1,activation = 'sigmoid'))
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics = 'AUC')
        model.fit(X_train,y_train,validation_data=(X_test, y_test), epochs = 5, batch_size = 32)
        models[pitch] = [pipe, model]
    return models

model_ball(last, first)
st.write('Loading pitcher')
pitch = st.selectbox('Pitch',list(load_pitcher(last,first))[1])
pitch_no = st.number_input('Pitch Number',0,200)

def enter_pitch(last,first,pitch,pitch_no):
    pitcher, pitches = load_pitcher(last,first)
    new = pd.DataFrame(pitcher.loc[(pitcher['pitch_type'] == pitch) & \
                                   (pitcher['pitch_no'] > pitch_no - 5) & \
                                   (pitcher['pitch_no'] < pitch_no + 5), :].agg('mean').copy())
    new = new.T.drop(columns = ['swing','ball','plate_x','plate_z','sz_top','sz_bot'])
    pipe = model_ball(last,first)[pitch][0]
    model = model_ball(last, first)[pitch][1]
    new = pipe.transform(new)
    new = pd.DataFrame(new, columns = pipe.get_feature_names_out())
    st.write(f'Chance of Ball with {pitch} is {model.predict(new)}')

enter_pitch(last,first,pitch,pitch_no)