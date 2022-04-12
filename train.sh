python train.py --emotion happy --audio_hidden 4 --video_hidden 16 --text_hidden 128 --audio_dropout 0.3 --video_dropout 0.1 --text_dropout 0.5 --factor_learning_rate 0.001 --learning_rate 0.001 --rank 1 --batch_size 128 --weight_decay 0.002

python train.py --emotion sad --audio_hidden 8 --video_hidden 4 --text_hidden 128 --audio_dropout 0 --video_dropout 0 --text_dropout 0 --factor_learning_rate 0.0005 --learning_rate 0.003 --rank 4 --batch_size 64 --weight_decay 0.002

python train.py --emotion angry --audio_hidden 8 --video_hidden 4 --text_hidden 64 --audio_dropout 0.3 --video_dropout 0.1 --text_dropout 0.15 --factor_learning_rate 0.003 --learning_rate 0.0005 --rank 8 --batch_size 64 --weight_decay 0.001

python train.py --emotion neutral --audio_hidden 32 --video_hidden 8 --text_hidden 64 --audio_dropout 0.2 --video_dropout 0.5 --text_dropout 0.2 --factor_learning_rate 0.003 --learning_rate 0.0005 --rank 16 --batch_size 16 --weight_decay 0.002
