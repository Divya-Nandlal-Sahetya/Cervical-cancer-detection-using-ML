#%cd /content/drive/MyDrive/EE_660_Project_(Divya&Maitreyee)/Code
from imports import *

def data_to_csv(balanced1, balanced2, balanced3, balanced4):
  balanced1.to_csv("./data/df1.csv", encoding='utf-8', index=False)
  balanced2.to_csv("./data/df2.csv", encoding='utf-8', index=False)
  balanced3.to_csv("./data/df3.csv", encoding='utf-8', index=False)
  balanced4.to_csv("./data/df4.csv", encoding='utf-8', index=False)
  
def test_data_to_csv(balanced1, balanced2, balanced3, balanced4):
  balanced1.to_csv("./data/test1.csv", encoding='utf-8', index=False)
  balanced2.to_csv("./data/test2.csv", encoding='utf-8', index=False)
  balanced3.to_csv("./data/test3.csv", encoding='utf-8', index=False)
  balanced4.to_csv("./data/test4.csv", encoding='utf-8', index=False)
  
# def train_data_to_csv(balanced1, balanced2, balanced3, balanced4):
#   balanced1.to_csv("./data/train1.csv", encoding='utf-8', index=False)
#   balanced2.to_csv("./data/train2.csv", encoding='utf-8', index=False)
#   balanced3.to_csv("./data/train3.csv", encoding='utf-8', index=False)
#   balanced4.to_csv("./data/train4.csv", encoding='utf-8', index=False)
  
def train_labeled_data_to_csv(data,percent):
    for i,(X,y) in enumerate(data):
        X = X.reset_index()
        y = y.reset_index()
        display(X)
        display(y)
        df = X.join(y)
        display(df)
        df.to_csv("./data/train_data_{}_{}.csv".format(i,percent), encoding='utf-8', index=False)

def get_dataframe():
  df1 = pd.read_csv("./data/df1.csv")
  df2 = pd.read_csv("./data/df2.csv")
  df3 = pd.read_csv("./data/df3.csv")
  df4 = pd.read_csv("./data/df4.csv")
  return df1, df2, df3, df4
  
def get_test_dataframe():
  test1 = pd.read_csv("./data/test1.csv")
  test2 = pd.read_csv("./data/test2.csv")
  test3 = pd.read_csv("./data/test3.csv")
  test4 = pd.read_csv("./data/test4.csv")
  return test1, test2, test3, test4
  
def normalise_data(X):                  
    # First regularize data b/w 1 and 2
    scaler = MinMaxScaler(feature_range=(1, 2))
    pipeline = Pipeline(steps=[('s', scaler)])
    X = pipeline.fit_transform(X)
    X = pd.DataFrame(X)
    return X

def get_X_and_y(df1, df2, df3, df4,normalize=False):
  X1 = df1.drop(["Hinselmann"], axis=1)
  y1 = df1["Hinselmann"].to_frame()
  X2 = df2.drop(["Schiller"], axis=1)
  y2 = df2["Schiller"].to_frame()
  X3 = df3.drop(["Citology"], axis=1)
  y3 = df3["Citology"].to_frame()
  X4 = df4.drop(["Biopsy"], axis=1)
  y4 = df4["Biopsy"].to_frame()

  y1.columns.name = 'Hinselmann'
  y2.columns.name = 'Schiller'
  y3.columns.name = 'Citology'
  y4.columns.name = 'Biopsy'
  
  if normalize:
      X1 = normalise_data(X1)
      X2 = normalise_data(X2)
      X3 = normalise_data(X3)
      X4 = normalise_data(X4)

  data_tuple = [(X1, y1), (X2, y2), (X3, y3), (X4, y4)]
  targets = [y1.columns.name, y2.columns.name, y3.columns.name, y4.columns.name]

  return data_tuple, targets, X1, y1, X2, y2, X3, y3, X4, y4
  
def get_train_test_split(data_tuple):
  train_test_tuple = []
  for X, y in data_tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)
    train_test_tuple.append((X_train, y_train, X_test, y_test))
  return train_test_tuple

def get_classifiers():
  knn_clf = KNeighborsClassifier(n_neighbors=2)
  log_clf = LogisticRegression()
  svm_clf = SVC(C=1, probability=True) #Pipeline([ ("pre", preprocessing.StandardScaler()), ("classifier", SVC(C=1, probability=True, random_state=42))])
  dt_clf = DecisionTreeClassifier()
  rf_clf = RandomForestClassifier()
#   xgb_clf = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=0.4, random_state=42)
  adb_clf = AdaBoostClassifier()

  clfs = [knn_clf, log_clf, svm_clf, dt_clf, rf_clf, adb_clf]
  model_log = [clf.__class__.__name__ for clf in clfs]
  return clfs,model_log

def get_hyper_parameters():
  params1={'n_neighbors':list(range(2,31))}
  params2={'penalty': ['l1','l2'], 'C':np.logspace(-3,3,7)}
  params3={"kernel":["rbf", "poly"], "gamma": ["auto", "scale"], "degree":range(1,6,1)}
  params4={'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
  params5={'n_estimators': [200, 500],
          'max_features': ['auto', 'sqrt', 'log2'],
          'max_depth' : [4,5,6,7,8],
          'criterion' :['gini', 'entropy']}
#   params6={'max_depth': range (2, 10, 1),'n_estimators': range(60, 220, 40),
        #   'learning_rate': [0.1, 0.01, 0.05]}
  params6={'base_estimator':[DecisionTreeClassifier(random_state=42)],
          'n_estimators':[10,50,250,1000],
          'learning_rate':[0.01,0.1]}
  parameters_list = [params1, params2, params3, params4, params5, params6]
  return parameters_list

def hyper_parameter_tuning(clfs,parameters_list,model_log,X,y):
  best_parameters = {}
  df = pd.DataFrame(columns = ['accuracy'])
  for i in range(len(clfs)):
      cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=True)
      Grid=GridSearchCV(estimator=clfs[i], param_grid=parameters_list[i],n_jobs=-1, cv=cv, verbose=3).fit(X, y)
      best_parameters[model_log[i]] = Grid.best_params_
      accuracy = Grid.best_score_ *100
      df = df.append({'accuracy' : accuracy},ignore_index=True)
      print("Accuracy for our training dataset with tuning is {}: {:.2f}% for \
      params {}".format(clfs[i],accuracy,best_parameters[model_log[i]]) )
      print("Classifier = {} M_train = {}".format(clfs[i], len(Grid.cv_results_['params'])))
  display(df)
  return best_parameters

# wrapper for  hyper_parameter_tuning
def get_best_params(data,percentage,clfs,parameters_list,model_log):
    best_params = []
    path = "./param/best_params_{}.pkl".format(percentage)
    i = 1
    if not os.path.isfile(path): 
        for X,y in data:
          print("Test = {}".format(i))
          best_params.append(hyper_parameter_tuning(clfs,parameters_list,model_log,X,y))
          i +=1
        with open(path,'wb') as f:
          pickle.dump(best_params,f)
    else:
      with open(path, 'rb') as f:
        best_params = pickle.load(f)
    return best_params

# sl to ssl  
def convert_sl_to_ssl_data(X_train,y_train,percentage):
  # split X_train and y_train into labelled and unlabelled
  X_l, X_u, y_l, y_u = train_test_split(X_train, y_train, test_size=percentage, random_state=42,stratify=y_train)
  # logic to convert labeleled points to unlabeled using stratified sampling
  y_u[y_u.columns] = -1 # set unlabeled targets to -1
  
  y_percentage = pd.concat([y_l, y_u], axis=0) # concatenate labeled and unlabeled labels
  X_percentage = pd.concat([X_l, X_u], axis=0) # conactenate labeled and unlabled features
  
  return X_percentage, y_percentage

def convert_for_all_tests(train_data,percentage):
  data = []
  for i,value in enumerate(train_data):
    X,y = value
    ### get percentage amount of unalabelled labelled data  + labelled data
    X_percentage, y_percentage = convert_sl_to_ssl_data(X,y,percentage)
    data.append((X_percentage, y_percentage))
  return data

def get_labelled_data(train_data):
  data = []
  for i,value in enumerate(train_data):
    X,y = value
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    joint = X.join(y)
    df_train_labeled = joint.loc[joint[y.columns[0]] != -1]
    X = df_train_labeled.iloc[:,:-1]
    y = df_train_labeled.iloc[:,-1:]
    data.append((X,y))
  return data
  
#   References:

# *   Kaggle
# 1.   https://www.kaggle.com/code/saqibsarwarkhan/cervical-cancer-risk-analysis
# 2.   https://www.kaggle.com/datasets/loveall/cervical-cancer-risk-classification


# *   GitHub
# 1.   https://github.com/topics/cervical-cancer
# 2.   https://github.com/avivace/cervical-cancer/blob/master/report.pdf


# *   Datahub.io: https://datahub.io/machine-learning/cervical-cancer
