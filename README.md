Data Format:

Each line in the document should look like: text + '\t' + label

To run the experiment, change the settings in main.sh, and then run the script. The accuracy result will be shown as training goes.

    --bert_path : pretrained bert model path
    --bert_vocab : bert vocabulary file path
    --train_data : training data path
    --dev_data : dev data path
    --max_len : max len of the input sequence
    --batch_size : batch size
    --lr : learning rate
    --dropout : dropout ratio
    --number_class : number of classes
    --number_epoch : epoch number
    --gpu_id : which GPU to use
    --print_every : how many batches to print one temporary result
    --fine_tune
    --model_save_path : model save path

To run inference, use `inference.sh`:

``` 
    --max_len : max len of the input sequence
    --ckpt_path : model save path
    --test_data : data to evaluate (each line is text)
    --out_path : output result path
    --gpu_id : which GPU to use
```

