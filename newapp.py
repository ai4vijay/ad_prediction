import streamlit as st
import pandas as pd
import pickle 

st.title("""Predicting If a Mobile Ad will be Clicked or Not!!""")
st.write("Description: This is app is basically to help you predict if a mobile ad will be clicked or not based on the given datasets")
st.write("How to Use : Please upload csv file with all the input details :")

def prediction(inputdf):
	ctr_model = pickle.load(open('ctrmodel.pkl','rb'))
	x=ctr_model.predict(inputdf)
	if x[0] < 0.2:
		return 'The Ad will likely to be not clicked'
	else:
		return 'The Ad is likely to be CLICKED'
"""@st.cache
def convert_df(df):
	return df.to_csv().encode('utf-8')
	
datafile = st.file_uploader("Upload CSV FILE ", type = ["csv"])"""

if datafile is not None:
	inputdata = pd.read_csv(datafile)
	st.dataframe(inputdata)
csv = convert_df(datafile)
st.download_button(lable='download sample'
)
if st.button("Predict"):
	pred=prediction(inputdata)
	st.write('Here is the prediction based on the input dataset')
	st.success(pred)