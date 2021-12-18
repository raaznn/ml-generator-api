from process import feature_data,cate_to_num,Data_normalized,create_dummies
def test_preprocessing(df1):
  input_data=df1.dropna(axis=1)
  feature_data_=feature_data(input_data)
  cate_col_lis=cate_to_num(input_data,feature_data_)
  coyp_cate_col=cate_col_lis.copy()
  # print(feature_data_)
  if len(cate_col_lis)==0:
      norm_df=Data_normalized(input_data)
  else:
      dummies_df=create_dummies(input_data,coyp_cate_col)
      norm_df=Data_normalized(dummies_df)
  
  length=len(norm_df)
  model_test_data=norm_df[length-1]
  return model_test_data
