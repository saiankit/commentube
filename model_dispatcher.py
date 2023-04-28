


models = {
     "": tree.DecisionTreeClassifier(
         criterion="gini"
     ),
     "decision_tree_entropy": tree.DecisionTreeClassifier(
         criterion='entropy'
     ),
     "rf": ensemble.RandomForestClassifier(),
     "log_reg": linear_model.LogisticRegression(),
     "svc": svm.SVC(C=10, gamma=0.001, kernel="rbf")
}
