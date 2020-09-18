python inference.py \
    --max_len 128 \
    --ckpt_path ./ckpt/kvbert_epoch_3 \
    --test_data ./data/KvPI_test.txt \
    --out_path ./test_prediction.txt \
    --gpu_id 0

python f1_acc.py --target data/KvPI_test.txt --pred test_prediction.txt
