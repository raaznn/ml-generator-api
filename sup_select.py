from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
model_params_classification = {
    'svm': {
        'model':SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}
model_params_prediction = {
  'GradientBoostingRegressor' : {
      'model': GradientBoostingRegressor(),
    'params' : {
        'learning_rate': [0.01,0.02]
  }
  },
  'DecisionTreeRegressor':{
    'model':DecisionTreeRegressor(),
      'params' : {
        'max_depth': [5,10,15]
  }
  },
  'random_forest': {
  'model': RandomForestRegressor(),
  'params' : {
    'n_estimators': [1,5,10]
  },
}
}
def gridesearch(x,y_train,model_dict):
  scores=[]
  for model_name, mp in model_dict.items():
    clf =  GridSearchCV(mp['model'], mp['params'],cv=2,return_train_score=False)
    clf.fit(x,y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
            })
  return scores

def train_machine(x,y,target_type):
    X_train, X_test, y_train, y_test = train_test_split( x, y ,test_size=0.15, random_state=42)
    if target_type=="prediction":
      result=gridesearch(X_train,y_train,model_params_prediction)
      result_df = pd.DataFrame(result,columns=['model','best_score','best_params'])
      return_result=result_df.sort_values(by='best_score',ascending=False,ignore_index=True).loc[0]
      optimized_model =model_params_prediction[return_result.model]['model']
      reg=optimized_model.fit(X_train,y_train)
      return reg

    else:
      result=gridesearch(X_train,y_train,model_params_classification)
      result_df = pd.DataFrame(result,columns=['model','best_score','best_params'])
      all_model=result_df.sort_values(by='best_score',ascending=False,ignore_index=True)
      return_result=all_model.loc[0]
      optimized_model =model_params_classification[return_result.model]['model']
      reg=optimized_model.fit(X_train,y_train)
      return reg

if __name__=="__main__":
    print("Requried libarary for supervised model are loaded")
