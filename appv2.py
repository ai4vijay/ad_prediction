import streamlit as st
import pandas as pd
import pickle 

st.title("""Predicting If a Mobile Ad will be Clicked or Not!!""")
st.write("Description: This is app is basically to help you predict if a mobile ad will be clicked or not based on the given datasets")
st.write("How to Use : Please upload csv file with all the input details:")

def prediction(inputdf,model):
	#ctr_model = pickle.load(open('ctrmodel.pkl','rb'))
	x=model.predict(inputdf)
	#return [true if ele > 0.2 else false ele for ele in x ]
	thresold = .2
	return [ 1 if ele > thresold else 0 for ele in x]
		
datafile = st.file_uploader("Upload CSV FILE ", type = ["csv"])
if datafile is not None:
	inputdata = pd.read_csv(datafile)
	st.dataframe(inputdata)

if st.checkbox('Version2'):
	ctr_model = pickle.load(open('ctr_model_v2.pkl','rb'))
else:
	ctr_model = pickle.load(open('ctrmodel.pkl','rb'))
	
if st.button("Predict"):
	pred=prediction(inputdata,ctr_model)
	st.write('Here is the prediction based on the input dataset')
	st.success(pred)