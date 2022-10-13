import streamlit as st
import pandas as pd
import pickle 
from pandas.api.types import is_string_dtype
from pandas.api.types import is_integer_dtype
import numpy as np

st.title("""Predicting If a Mobile Ad will be Clicked or Not!!""")
st.write("Description: This is app is basically to help you predict if a mobile ad will be clicked or not based on the given datasets")
st.write("How to Use : Please upload csv file with all the input details , here are the input details that must be present in the csv file :")

def prediction(inputdf):
	ctr_model = pickle.load(open('ctrmodel.pkl','rb'))
	x=ctr_model.predict(inputdf)
	if x[0] < 0.2:
		return 'The Ad will likely to be not clicked'
	else:
		return 'The Ad is likely to be CLICKED'
#column data types
types_df = {
    'id': np.dtype(int),
    'click': np.dtype(int),
    'hour': np.dtype(int),
    'C1': np.dtype(int),
    'banner_pos': np.dtype(int),
    'site_id': np.dtype(str),
    'site_domain': np.dtype(str), 
    'site_category': np.dtype(str),
    'app_id': np.dtype(str),
    'app_domain': np.dtype(str),
    'app_category': np.dtype(str),
    'device_id': np.dtype(str),
    'device_ip': np.dtype(str),
    'device_model': np.dtype(str),
    'device_type': np.dtype(int),
    'device_conn_type': np.dtype(int),
    'C14': np.dtype(int),
    'C15': np.dtype(int),
    'C16': np.dtype(int),
    'C17': np.dtype(int),
    'C18': np.dtype(int),
    'C19': np.dtype(int),
    'C20': np.dtype(int),
    'C21':np.dtype(int)
}

def convert_obj_to_int(self):
    
    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    for index in range(0,len(object_list_columns)):
        if object_list_dtypes[index] == object :
            self[object_list_columns[index]] = self[object_list_columns[index]].map( lambda  x: hash(x))
            self.drop([object_list_columns[index]],inplace=True,axis=1)
    return self

def inputclean(inputdata):
	df=inputdata
	#df = pd.DataFrame()
	st.dataframe(inputdata)
	if st.checkbox("replace"):
		mydf = st.dataframe(df)
		columns = st.selectbox("select columns" , df.columns)
		#old_values = st.multiselect("current values", list(df[columns].unique(),list(df[columns].unique()))
		old_values = st.multiselect("Current Values",list(df[columns].unique()),list(df[columns].unique()))
		with st.form(key='my_form'):
			col1,col2 = st.columns(2)
			st_input = st.number_input if is_integer_dtype(df[columns]) else st.text_input
			#st_input= st.text_input
			with col1:
				old_val = st_input("old value")
			with col2:
				new_val = st_input("new value")
			if st.form_submit_button("Submit"):
				df[columns]=df[columns].replace(old_val,new_val)
				st.success("{} replace with {} successfully ".format(old_val,new_val))
				mydf.add_rows(df)
				df = convert_obj_to_int(df)
				return df
def convert_df(df):
	return df.to_csv().encode('utf-8')
datafile = st.file_uploader("Upload CSV FILE ", type = ["csv"])
csv = convert_df(datafile)
st.download_button('sample',csv)
if datafile is None:
	inputdata = pd.read_csv("newtestdf.csv")
	mydf = inputclean(inputdata)
else:
	inputdata = pd.read_csv(datafile)
	mydf=inputclean(inputdata)
	
#st.download_button(label ="download sample",data=datafile,filename ='sample_input.csv',mime='text/csv',)

if st.button("Predict"):
	pred=prediction(mydf)
	st.write('Here is the prediction based on the input dataset')
	st.success(pred)