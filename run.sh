
base_dir="/home/liangliu/github/llnlp/data"
bert_dir="/home/liangliu/github/chinese_L-12_H-768_A-12/"

python3 train_test.py --lr 0.0005 \
		--num_epochs 5 \
		--batch_size 32 \
		--dropout_rate 0.1 \
		--train_path $base_dir/train.conll \
		--test_path  $base_dir/dev.conll \
		--vocab_file ${bert_dir}/vocab.txt \
		--tag_mapping_file ${base_dir}/tags.txt \
		--bert_config ${bert_dir}/bert_config.json
