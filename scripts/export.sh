python ../tools/export.py --emotion happy --audio_hidden 4 --video_hidden 16 --text_hidden 128 --audio_dropout 0.3 --video_dropout 0.1 --text_dropout 0.5 --rank 1 --data_path ../data/iemocap.pkl
python ../tools/export.py --emotion sad --audio_hidden 8 --video_hidden 4 --text_hidden 128 --audio_dropout 0 --video_dropout 0 --text_dropout 0 --rank 4 --data_path ../data/iemocap.pkl
python ../tools/export.py --emotion angry --audio_hidden 8 --video_hidden 4 --text_hidden 64 --audio_dropout 0.3 --video_dropout 0.1 --text_dropout 0.15 --rank 8 --data_path ../data/iemocap.pkl
python ../tools/export.py --emotion neutral --audio_hidden 32 --video_hidden 8 --text_hidden 64 --audio_dropout 0.2 --video_dropout 0.5 --text_dropout 0.2 --rank 16 --data_path ../data/iemocap.pkl
