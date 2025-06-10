mkdir data
tar zxvf data/sample_train.tar.gz -C data
tar zxvf data/sample_test.tar.gz -C data
python process_public_data.py
mv data/ctr_cvr.train data/train.csv
mv data/ctr_cvr.test data/test.csv
