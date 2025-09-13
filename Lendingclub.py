import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df=pd.read_csv('loan.csv')


print(df[df['loan_status']=='Charged Off'])


plt.figure(figsize=(40,20))


sns.boxplot(x=df['loan_status'],y=df['loan_amnt'],data=df)

plt.tight_layout()

df.head()

df=df.drop('id',axis=1)


df=df.drop('member_id',axis=1)

df.columns

df.isnull().sum()

df.info()

df=df.drop(['settlement_status','settlement_date','settlement_amount','settlement_percentage','settlement_term'],axis=1)

df.isnull().sum()

df=df.drop(['hardship_payoff_balance_amount','hardship_last_payment_amount','hardship_last_payment_amount','debt_settlement_flag_date'],axis=1)

df.head()

c=df.isnull().sum()
c=c[c>1000000].index

df.info()

df=df.drop(columns=c,axis=1)

df.isnull().sum().sort_values(ascending=False).head(20)

c=df.isnull().sum()

df=df.drop(['mths_since_rcnt_il','all_util','open_acc_6m','inq_last_12m','total_cu_tl','total_bal_il','open_rv_12m','open_rv_24m','open_act_il','max_bal_bc','open_il_12m','inq_fi','open_il_24m'],axis=1)

df.isnull().sum().sort_values(ascending=False).head(20)

df['emp_title']

df=df.drop('emp_title',axis=1)

df.isnull().sum().sort_values(ascending=False).head(20)

df['emp_length'].fillna('Other',inplace=True)

df['emp_length']=df['emp_length'].apply(lambda x:x[0:1])

df.columns

df.isnull().sum().sort_values(ascending=False).head(20)

df['mths_since_recent_inq'].fillna(0,inplace=True)

df.isnull().sum().sort_values(ascending=False).head(20)

df['num_tl_120dpd_2m'].fillna(0,inplace=True)

df['mo_sin_old_il_acct'].fillna(0,inplace=True)

df.corrwith(df['bc_util'],numeric_only=True).sort_values(ascending=False).head(5)

df['bc_util'].fillna(0,inplace=True)

df.info()

df['grade']

df['sub_grade']

df=df.drop('sub_grade',axis=1)

df['home_ownership']

df['verification_status']

df=df.drop('issue_d',axis=1)

df['loan_status']

df['pymnt_plan'].nunique()

df.select_dtypes(include='object').columns.tolist()

df['purpose'].nunique()

df=df.drop('title',axis=1)

df['zip_code']=df['zip_code'].apply(lambda x: str(x)[0:3])

df['zip_code']

df=df.drop('addr_state',axis=1)

df['earliest_cr_line']=df['earliest_cr_line'].apply(lambda x:str(x)[-1:-5])

df=df.drop('earliest_cr_line',axis=1)

df.select_dtypes(include='object').columns.tolist()

df['initial_list_status'].nunique()

df['last_pymnt_d']=df['last_pymnt_d'].apply(lambda x:str(x)[4:])

df['last_pymnt_d']

df['last_credit_pull_d']=df['last_credit_pull_d'].apply(lambda x:str(x)[4:])

df['last_credit_pull_d']

df['application_type'].nunique()

df['hardship_flag'].nunique()

df['disbursement_method'].nunique()

df['debt_settlement_flag'].nunique()

df.select_dtypes(include='object').columns.tolist()

df=df.drop(['last_pymnt_d','last_credit_pull_d','application_type','hardship_flag','disbursement_method'],axis=1)

df['grade']

df.head()

df.columns

df['bc_util'].isnull().sum()

df=df.drop([
    'zip_code',
    'policy_code',
    'total_pymnt',
    'total_pymnt_inv',
    'total_rec_prncp',
    'total_rec_int',
    'total_rec_late_fee',
    'recoveries',
    'collection_recovery_fee',
    'last_pymnt_amnt',
    'debt_settlement_flag',
    'pymnt_plan'
],axis=1)

df=df.drop([
    'purpose',                # medium-cardinality categorical
    'initial_list_status',     # usually only 1-2 unique values
    'mths_since_recent_bc',    # mostly NaN
    'mths_since_recent_inq',   # mostly NaN
    'num_tl_120dpd_2m',        # rare events, low predictive value
    'num_tl_90g_dpd_24m',      # rare events, low predictive value
    'tax_liens'                # very few non-zero values
],axis=1)

df.columns

df.select_dtypes(include='object').columns.tolist()

cols=['term',
 'grade',
 'emp_length',
 'home_ownership',
 'verification_status',
 'loan_status']

dum=pd.get_dummies(df[cols],drop_first=True,dtype=int)

df=pd.concat([df,dum],axis=1)

df=df.drop(['term',
 'grade',
 'emp_length',
 'home_ownership',
 'verification_status',
 'loan_status']
,axis=1)

df.head()

df.select_dtypes(include='object').columns.tolist()

y=df[['loan_status_Current','loan_status_Default','loan_status_Does not meet the credit policy. Status:Charged Off','loan_status_Does not meet the credit policy. Status:Fully Paid','loan_status_Fully Paid','loan_status_In Grace Period','loan_status_Late (16-30 days)','loan_status_Late (31-120 days)']]



X=df.drop(['loan_status_Current','loan_status_Default','loan_status_Does not meet the credit policy. Status:Charged Off','loan_status_Does not meet the credit policy. Status:Fully Paid','loan_status_Fully Paid','loan_status_In Grace Period','loan_status_Late (16-30 days)','loan_status_Late (31-120 days)'],axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=25,n_jobs=-1)

rfc.fit(X_train,y_train)


predictions=rfc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

y_test_labels=y_test.idxmax(axis=1)

classes=y_test.columns
pred=np.argmax(predictions,axis=1)

pred1=classes[pred]

print(confusion_matrix(y_test_labels,pred1))

print('\n')

print(classification_report(y_test_labels,pred1))


df.isnull().sum().sort_values(ascending=False).head(30)

df.fillna(0,inplace=True)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

y_train.shape[1]

from sklearn.preprocessing import StandardScaler

y=df[['loan_status_Current','loan_status_Default','loan_status_Does not meet the credit policy. Status:Charged Off','loan_status_Does not meet the credit policy. Status:Fully Paid','loan_status_Fully Paid','loan_status_In Grace Period','loan_status_Late (16-30 days)','loan_status_Late (31-120 days)']]



X=df.drop(['loan_status_Current','loan_status_Default','loan_status_Does not meet the credit policy. Status:Charged Off','loan_status_Does not meet the credit policy. Status:Fully Paid','loan_status_Fully Paid','loan_status_In Grace Period','loan_status_Late (16-30 days)','loan_status_Late (31-120 days)'],axis=1)






X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)



s=StandardScaler()

X_train=s.fit_transform(X_train)

X_test=s.transform(X_test)

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=1e-6)

model=Sequential()

model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))


model.add(Dense(128,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(32,activation='relu'))

model.add(Dropout(0.3))


model.add(Dense(8,activation='softmax'))

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])

Early=EarlyStopping(monitor='val_accuracy',patience=15,verbose=1,mode='max',restore_best_weights=True)

model.fit(X_train,y_train,epochs=500,verbose=1,callbacks=[Early],validation_data=(X_test,y_test),batch_size=128)

X_train.min(), X_train.max()

print(np.isnan(X_train).sum(), np.isinf(X_train).sum())

y_train.sum(axis=0)

y_train.shape

model.evaluate(X_train,y_train)