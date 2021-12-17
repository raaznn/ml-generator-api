from preprocessing import feature_data,cate_to_num,Data_normalized
from preprocessing import create_dummies
from sup_select import train_machine
from unsup_select import unsupervised_model
import pandas as pd

def estimate(df,target_col):    
    feature_data_=feature_data(df)
    cate_col_lis=cate_to_num(df,feature_data_)
    coyp_cate_col=cate_col_lis.copy()
    if target_col=="":
        input_data=df
        if len(cate_col_lis)>0:
            dummies_df=create_dummies(input_data,coyp_cate_col)
            norm_df=Data_normalized(dummies_df)
        else:
            norm_df=Data_normalized(input_data)

        model=unsupervised_model(norm_df) 
        return model
    else:
        input_data=df.drop(columns=target_col)
        if len(cate_col_lis)==0:
            norm_df=Data_normalized(input_data)
        elif (len(cate_col_lis)>0) and (target_col in cate_col_lis): 
            coyp_cate_col.remove(target_col)
            if len(coyp_cate_col)==0:
                norm_df=Data_normalized(input_data)
            dummies_df=create_dummies(input_data,coyp_cate_col)
            norm_df=Data_normalized(dummies_df)
        elif (len(cate_col_lis)>0) and (target_col not in cate_col_lis):
            dummies_df=create_dummies(input_data,coyp_cate_col)
            norm_df=Data_normalized(dummies_df)
    if target_col in cate_col_lis:
        return_model,best_mdl_dict,all_mdl_dict=train_machine(norm_df,df[target_col],"classification")
    else:
        return_model,best_mdl_dict,all_mdl_dict=train_machine(norm_df,df[target_col],"prediction")
    return return_model,best_mdl_dict,all_mdl_dict

if __name__=='__main__':
    print("process are alright")