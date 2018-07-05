套件版本：
python==3.5.2 keras==2.1.0
numpy==1.14.2
scikit-learn==0.19.1
tensorflow-gpu==1.4.0
tensorflow-tensorboard==0.4.0
xgboost ==0.72

test指令： bash test.sh $1 
($1 為predict.csv所存的位置，如 ./predict.csv)
training指令： bash train.sh 
(會在./model/ 中產生model)