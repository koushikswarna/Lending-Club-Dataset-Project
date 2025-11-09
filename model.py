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
