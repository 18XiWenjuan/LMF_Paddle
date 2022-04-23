python ../tools/eval.py --emotion happy --audio_hidden 4 --video_hidden 16 --text_hidden 128 --audio_dropout 0.3 --video_dropout 0.1 --text_dropout 0.5 --rank 1 --check_path ../check/happy.pkl
python ../tools/eval.py --emotion sad --audio_hidden 8 --video_hidden 4 --text_hidden 128 --audio_dropout 0 --video_dropout 0 --text_dropout 0 --rank 4 --check_path ../check/sad.pkl
python ../tools/eval.py --emotion angry --audio_hidden 8 --video_hidden 4 --text_hidden 64 --audio_dropout 0.3 --video_dropout 0.1 --text_dropout 0.15 --rank 8 --check_path ../check/angry.pkl
python ../tools/eval.py --emotion neutral --audio_hidden 32 --video_hidden 8 --text_hidden 64 --audio_dropout 0.2 --video_dropout 0.5 --text_dropout 0.2 --rank 16 --check_path ../check/neutral.pkl
