parameters = {'n_neightbors' : [1, 2. 3. 4. 5,6 ,7, 8, 9 .10 ],
'algorithm': ['auto,' 'ball_tree', 'kd_tree,' 'brute']
KNN = KNeighborsClassifier()
gscv=GridsearchCV(KNN,parameter,scoring="accuracy", cv=10)
KNN_cv=gscv.fit(X_train,y_train) 


print("Accuracy", KNN_cv.score(X_test,y_test))
print("tuned hyperparameters :(best parameters)", KNN_cv.best_params_)
print ('accuracy :",KNN_cv.best_score_)

