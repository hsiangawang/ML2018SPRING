wget https://www.dropbox.com/s/mfo9hi1uo2c4v41/final_model.h5
mv final_model.h5 model/

wget --header 'Host: doc-0k-3s-docs.googleusercontent.com' --user-agent 'Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:61.0) Gecko/20100101 Firefox/61.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: zh-TW,zh;q=0.8,en-US;q=0.5,en;q=0.3' --referer 'https://drive.google.com/drive/my-drive' --header 'Cookie: AUTH_v9lb9au55hd5je1h8pbbnrahluk2a4bc_nonce=ttlucnt57div0' --header 'Upgrade-Insecure-Requests: 1' 'https://doc-0k-3s-docs.googleusercontent.com/docs/securesc/nvtm58vsel2ncaua76ug6sfjadj7q9kp/2htqnceaorsh2lpvnr6qhtc01qhllon1/1530784800000/14169530281466620780/14169530281466620780/1gfnNEE9Ds0b3br_uGs8MIjYT1HugPcbO?e=download&nonce=ttlucnt57div0&user=14169530281466620780&hash=flrq5mnh4ra81dmu6v520rivi2kktplr' --output-document 'data.zip'
mv data.zip data/
cd data
unzip data.zip
cd ..



