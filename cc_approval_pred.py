import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
#from secret import access_key, secret_access_key
import joblib
import streamlit as st
import boto3
import tempfile
import json
import requests
from streamlit_lottie import st_lottie_spinner




train_original = pd.read_csv('https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/main/dataset/train.csv')

test_original = pd.read_csv('https://raw.githubusercontent.com/semasuka/Credit-card-approval-prediction-classification/main/dataset/test.csv')

full_data = pd.concat([train_original, test_original], axis=0)

full_data = full_data.sample(frac=1).reset_index(drop=True)


def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()



def value_cnt_norm_cal(df,feature):
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat



class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_outliers = ['Family member count','Income', 'Employment length']):
        self.feat_with_outliers = feat_with_outliers
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_outliers).issubset(df.columns)):
            # 25% quantile
            Q1 = df[self.feat_with_outliers].quantile(.25)
            # 75% quantile
            Q3 = df[self.feat_with_outliers].quantile(.75)
            IQR = Q3 - Q1
            # keep the data within 1.5 IQR
            df = df[~((df[self.feat_with_outliers] < (Q1 - 1.5 * IQR)) |(df[self.feat_with_outliers] > (Q3 + 1.5 * IQR))).any(axis=1)]
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class DropFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,feature_to_drop = ['ID','Has a mobile phone','Children count','Job title','Account age']):
        self.feature_to_drop = feature_to_drop
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop,axis=1,inplace=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days = ['Employment length', 'Age']):
        self.feat_with_days = feat_with_days
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if (set(self.feat_with_days).issubset(X.columns)):
            # convert days to absolute value
            X[['Employment length','Age']] = np.abs(X[['Employment length','Age']])
            return X
        else:
            print("One or more features are not in the dataframe")
            return X


class RetireeHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, df):
        return self
    def transform(self, df):
        if 'Employment length' in df.columns:
            # select rows with employment length is 365243 which corresponds to retirees
            df_ret_idx = df['Employment length'][df['Employment length'] == 365243].index
            # change 365243 to 0
            df.loc[df_ret_idx,'Employment length'] = 0
            return df
        else:
            print("Employment length is not in the dataframe")
            return df


class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_skewness=['Income','Age']):
        self.feat_with_skewness = feat_with_skewness
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_skewness).issubset(df.columns)):
            # Handle skewness with cubic root transformation
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_num_enc=['Has a work phone','Has a phone','Has an email']):
        self.feat_with_num_enc = feat_with_num_enc
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.feat_with_num_enc).issubset(df.columns)):
            # Change 0 to N and 1 to Y for all the features in feat_with_num_enc
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1:'Y',0:'N'})
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


class OneHotWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,one_hot_enc_ft = ['Gender', 'Marital status', 'Dwelling', 'Employment status', 'Has a car', 'Has a property', 'Has a work phone', 'Has a phone', 'Has an email']):
        self.one_hot_enc_ft = one_hot_enc_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.one_hot_enc_ft).issubset(df.columns)):
            # function to one hot encode the features in one_hot_enc_ft
            def one_hot_enc(df,one_hot_enc_ft):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_enc_ft])
                # get the result of the one hot encoding columns names
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)
                # change the array of the one hot encoding to a dataframe with the column names
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(),columns=feat_names_one_hot_enc,index=df.index)
                return df
            # function to concatenat the one hot encoded features with the rest of features that were not encoded
            def concat_with_rest(df,one_hot_enc_df,one_hot_enc_ft):
                # get the rest of the features
                rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                # concatenate the rest of the features with the one hot encoded features
                df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]],axis=1)
                return df_concat
            # one hot encoded dataframe
            one_hot_enc_df = one_hot_enc(df,self.one_hot_enc_ft)
            # returns the concatenated dataframe
            full_df_one_hot_enc = concat_with_rest(df,one_hot_enc_df,self.one_hot_enc_ft)
            return full_df_one_hot_enc
        else:
            print("One or more features are not in the dataframe")
            return df


class OrdinalFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,ordinal_enc_ft = ['Education level']):
        self.ordinal_enc_ft = ordinal_enc_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Education level' in df.columns:
            ordinal_enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
            return df
        else:
            print("Education level is not in the dataframe")
            return df

class MinMaxWithFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self,min_max_scaler_ft = ['Age', 'Income', 'Employment length']):
        self.min_max_scaler_ft = min_max_scaler_ft
    def fit(self,df):
        return self
    def transform(self,df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df

class OversampleSMOTE(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,df):
        return self
    def transform(self,df):
        if 'Is high risk' in df.columns:
            # SMOTE function to oversample the minority class to fix the imbalance data
            smote = SMOTE()
            X_bal, y_bal = smote.fit_resample(df.loc[:, df.columns != 'Is high risk'],df['Is high risk'].astype('int64'))
            df_bal = pd.concat([pd.DataFrame(X_bal),pd.DataFrame(y_bal)],axis=1)
            return df_bal
        else:
            print("Is high risk is not in the dataframe")
            return df


def full_pipeline(df):
    # Create the pipeline that will call all the class from OutlierRemoval to OversampleSMOTE in one go
    pipeline = Pipeline([
        ('outlier_remover', OutlierRemover()),
        ('feature_dropper', DropFeatures()),
        ('time_conversion_handler', TimeConversionHandler()),
        ('retiree_handler', RetireeHandler()),
        ('skewness_handler', SkewnessHandler()),
        ('binning_num_to_yn', BinningNumToYN()),
        ('one_hot_with_feat_names', OneHotWithFeatNames()),
        ('ordinal_feat_names', OrdinalFeatNames()),
        ('min_max_with_feat_names', MinMaxWithFeatNames()),
        ('oversample_smote', OversampleSMOTE())
    ])
    df_pipe_prep = pipeline.fit_transform(df)
    return df_pipe_prep




cc_train_prep = full_pipeline(train_copy)

st.write(cc_train_prep.head())


# def drop_least_useful_ft(prep_data,feat_list):
#     X_train_copy_prep_drop_ft = prep_data.drop(feat_list,axis=1)
#     return X_train_copy_prep_drop_ft



# ############################ Streamlit ############################

# st.write("""
# # Income Classification
# This app predicts if your income is high or low than $50000. Just fill in the following information and click on the Predict button.:
# """)

# # Age input slider
# st.write("""
# ## Age
# """)
# input_age = st.slider('Select your age', value=38, min_value=15, max_value=78, step=1)

# #Gender input
# st.write("""
# ## Gender
# """)
# input_gender = st.radio('Select you gender',['Male','Female'], index=0)


# # Workclass input dropdown
# st.write("""
# ## Workclass
# """)
# work_class_values = list(value_cnt_norm_cal(full_data,'workclass').index)
# work_class_key = ['Private sector', 'Self employed (not incorporated)', 'Local government', 'State government', 'Self employed (incorporated)', 'Without work', 'Never worked']
# work_class_dict = dict(zip(work_class_key,work_class_values))
# input_workclass_key = st.selectbox('Select your workclass', work_class_key)
# input_workclass_val = work_class_dict.get(input_workclass_key)


# # Education level input dropdown
# st.write("""
# ## Education level
# """)
# initial_edu_df = full_data[['education','educational-num']].drop_duplicates().sort_values(by='educational-num').reset_index(drop=True)
# edu_key = ['Pre-school', '1st to 4th grade', '5th to 6th grade', '7th to 8th grade', '9th grade', '10th grade', '11th grade', '12th grade no diploma', 'High school graduate', 'Some college', 'Associate degree (vocation)','Associate degree (academic)' ,'Bachelor\'s degree', 'Master\'s degree', 'Professional school', 'Doctorate degree']
# edu_df = pd.concat([initial_edu_df,pd.DataFrame(edu_key,columns=['education-letter'])],axis=1)
# edu_dict = edu_df.set_index('education-letter').to_dict()['educational-num']
# input_edu_key = st.selectbox('Select your highest education level', edu_df['education-letter'])
# input_edu_val = edu_dict.get(input_edu_key)
# input_education = edu_df.iloc[[input_edu_val-1]]['education'].values[0]


# # Marital status input dropdown
# st.write("""
# ## Marital status
# """)
# marital_status_values = list(value_cnt_norm_cal(full_data,'marital-status').index)
# marital_status_key = ['Married (civilian spouse)', 'Never married', 'Divorced', 'Separated', 'Widowed', 'Married (abscent spouse)', 'Married (armed forces spouse)']
# marital_status_dict = dict(zip(marital_status_key,marital_status_values))
# input_marital_status_key = st.selectbox('Select your marital status', marital_status_key)
# input_marital_status_val = marital_status_dict.get(input_marital_status_key)



# #Occupation input dropdown
# st.write("""
# ## Occupation
# """)
# occupation_values = list(value_cnt_norm_cal(full_data,'occupation').index)
# occupation_key = ['Craftman & repair', 'Professional specialty', 'Executive and managerial role', 'Administrative clerk','Sales', 'Other services', 'Machine operator & inspector', 'Transportation & moving', 'Handlers & cleaners', 'Farming & fishing', 'Technical support', 'Protective service', 'Private house service', 'Armed forces']
# occupation_dict = dict(zip(occupation_key,occupation_values))
# input_occupation_key = st.selectbox('Select your occupation', occupation_dict)
# input_occupation_val = occupation_dict.get(input_occupation_key)

# # Relationship input dropdown
# st.write("""
# ## Relationship
# """)
# relationship_values = list(value_cnt_norm_cal(full_data,'relationship').index)
# relationship_key = ['Husband', 'Not in a family', 'Own child', 'Not married','Wife', 'Other relative']
# relationship_dict = dict(zip(relationship_key,relationship_values))
# input_relationship_key = st.selectbox('Select the type of relationship', relationship_dict)
# input_relationship_val = relationship_dict.get(input_relationship_key)

# # Race input dropdown
# st.write("""
# ## Race
# """)
# race_values = list(value_cnt_norm_cal(full_data,'race').index)
# race_key = ['White', 'Black', 'Asian & pacific islander', 'American first nation','Other']
# race_dict = dict(zip(race_key,race_values))
# input_race_key = st.selectbox('Select your race', race_dict)
# input_race_val = race_dict.get(input_race_key)

# # Capital gain input
# st.write("""
# ## Capital gain
# """)
# input_capital_gain = st.text_input('Enter any capital gain amount',0,help='A capital gain is a profit from the sale of property or an investment.')


# # Capital gain input
# st.write("""
# ## Capital loss
# """)
# input_capital_loss = st.text_input('Enter any capital loss amount',0,help='A capital loss is a loss from the sale of property or an investment when sold for less than the price it was purchased for.')

# # Age input slider
# st.write("""
# ## Hours worked per week
# """)
# input_hours_worked = st.slider('Select the number of hours you work per week', value=40, min_value=0, max_value=110, step=1)

# # Country of residence input dropdown
# st.write("""
# ## Country of residence
# """)
# input_country = st.selectbox('Select your country of residence', gdp_data['native-country'].sort_values())
# gdp = gdp_grouping(input_country)

# st.markdown('##')
# st.markdown('##')
# # Button
# predict_bt = st.button('Predict')

# profile_to_predict = [input_age, input_workclass_val, 0, input_education, input_edu_val, input_marital_status_val, input_occupation_val, input_relationship_val, input_race_val,input_gender, float(input_capital_gain), float(input_capital_loss), input_hours_worked, input_country, gdp,-1.000]

# profile_to_predict_df = pd.DataFrame([profile_to_predict],columns=train_copy.columns)

# train_copy_with_profile_to_pred = pd.concat([train_copy,profile_to_predict_df],ignore_index=True)







# train_copy_prep = full_pipeline_fuc(train_copy)

# test_copy_prep = full_pipeline_fuc(test_copy)

# X_train_copy_prep = train_copy_prep.iloc[:,:-1]

# y_train_copy_prep = train_copy_prep.iloc[:,-1]


# X_test_copy_prep = test_copy_prep.iloc[:,:-1]


# y_test_copy_prep = test_copy_prep.iloc[:,-1]



# train_copy_with_profile_to_pred = full_pipeline_fuc(train_copy_with_profile_to_pred)

# profile_to_pred_prep = train_copy_with_profile_to_pred.iloc[-1:,:-1]








# rand_forest_least_pred = [
#     'occupation_Handlers-cleaners',
#     'workclass_Federal-gov',
#     'marital-status_Married-AF-spouse',
#     'race_Amer-Indian-Eskimo',
#     'occupation_Protective-serv',
#     'marital-status_Married-spouse-absent',
#     'race_Other',
#     'workclass_Without-pay',
#     'occupation_Armed-Forces',
#     'occupation_Priv-house-serv'
# ]



# profile_to_pred_prep_drop_ft = drop_least_useful_ft(profile_to_pred_prep,rand_forest_least_pred)

# st.markdown('##')
# st.markdown('##')


# #Animation function
# @st.experimental_memo
# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


# lottie_loading_an = load_lottieurl('https://assets3.lottiefiles.com/packages/lf20_szlepvdh.json')


# def make_prediction():
#     # connect to s3 bucket
#     client = boto3.client('s3', aws_access_key_id=st.secrets["access_key"],aws_secret_access_key=st.secrets["secret_access_key"]) # for s3 API keys when deployed on streamlit share
#     #client = boto3.client('s3', aws_access_key_id=access_key,aws_secret_access_key=secret_access_key) # for s3 API keys when deployed on locally

#     bucket_name = "incomepredbucket"
#     key = "rand_forest_clf.sav"

#     # load the model from s3 in a temporary file
#     with tempfile.TemporaryFile() as fp:
#         client.download_fileobj(Fileobj=fp, Bucket=bucket_name, Key=key)
#         fp.seek(0)
#         model = joblib.load(fp)

#     # prediction from the model on AWS S3
#     return model.predict(profile_to_pred_prep_drop_ft)

# if predict_bt:

#     with st_lottie_spinner(lottie_loading_an, quality='high', height='200px', width='200px'):
#         final_pred = make_prediction()
#     # if final_pred exists, then stop displaying the loading animation
#     if final_pred[0] == 1.0:
#         st.success('## You most likely make more than 50k')
#     else:
#         st.error('## You most likely make less than 50k')








