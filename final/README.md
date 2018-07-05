套件版本：<br />
python==3.5.2 <br />
keras==2.1.0 <br />
numpy==1.14.2 <br />
scikit-learn==0.19.1 <br />
scipy==0.19.0 <br />
tensorflow-gpu==1.4.0 <br />
tensorflow-tensorboard==0.4.0 <br />
xgboost ==0.72 <br />

data＆model 因為檔案太大所以透過另外下載 <br />
data & model指令：bash data.sh <br />
test指令： bash test.sh $1  <br />
($1 為predict.csv所存的位置，如 ./predict.csv) <br />
training指令： bash train.sh <br />
(會在./model/ 中產生model)<br />

