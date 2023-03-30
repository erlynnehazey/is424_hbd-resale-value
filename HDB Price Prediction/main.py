import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold as sk
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.preprocessing import StandardScaler, scale
from sklearn.metrics import accuracy_score as acs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("final_hdb_resale_prices.csv")

st.subheader("Please input the relevant features of the HDB flat to make a prediction:")
Storey_Range= st.number_input('Storey Range', 0, max(data["storey_range"]), 1)
remaining_lease_months = st.number_input('remaining_lease_months', 0, max(data["remaining_lease_months"]), 1)
nearest_mrt= st.number_input('Distance to nearest MRT', 0.0, max(data["Distance to nearest MRT"]), 1.0)
Distance_to_CBD	 = st.number_input('Distane to CBD', 0.0, max(data["Distance to CBD"]), 1.0)
isMatureEstate = st.number_input('isMatureEstate', 0, max(data["isMatureEstate"]), 1)
Floor_Area_Sqm = st.number_input('Floor Area Sqm', 0.0, max(data["floor_area_sqm"]), 1.0)
resale_application= st.number_input('resale application', 26436, disabled=True)
bto = st.number_input('no.of bto', 20440, disabled=True)

# scaler = MinMaxScaler()
# Storey_Range = scaler.fit_transform(Storey_Range)
# Floor_Area_Sqm  = scaler.fit_transform(Floor_Area_Sqm )
# remaining_lease_months = scaler.fit_transform(remaining_lease_months)
# input_Height = scaler.fit_transform(input_Height)
# input_Width = scaler.fit_transform(input_Width)
# isMatureEstate  = scaler.fit_transform(Storey_Range)

with open('best_model.pkl' , 'rb') as f:
    lr = pickle.load(f)


if st.button('Make Prediction'):
    with open('best_model.pkl' , 'rb') as f:
        lr = pickle.load(f)
        xx = lr.predict([[Storey_Range,Floor_Area_Sqm,bto,resale_application,remaining_lease_months,nearest_mrt,Distance_to_CBD,isMatureEstate]])
        # xx =  lr.predict([[0.375,0.098958,0,0,0.229833,0.242702,0.377101,1]])
        st.write('The predicted price is $'+str((xx[0]*Floor_Area_Sqm).round(2))) # multiple back by standard scalar formula  (x*sd + mean )