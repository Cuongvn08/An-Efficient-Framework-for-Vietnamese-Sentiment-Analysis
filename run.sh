
## aivivn
# rnn
python -u main.py -c config/aivivn/rnn/1.train.config
python -u main.py -c config/aivivn/rnn/2.train.config
python -u main.py -c config/aivivn/rnn/3.train.config
python -u main.py -c config/aivivn/rnn/4.train.config
# cnn
python -u main.py -c config/aivivn/cnn/1.train.config
# rcnn
python -u main.py -c config/aivivn/rcnn/1.train.config
# dpcnn
python -u main.py -c config/aivivn/dpcnn/1.train.config
# transformer
python -u main.py -c config/aivivn/transformer/1.train.config
# bert
python -u main_bert.py --PATH_STAR_RATING_TRAIN dataset/aivivn/train.csv --PATH_STAR_RATING_TEST dataset/aivivn/test.csv --OUTPUT_DIR output/aivivn/models/bert/
# bertpho
python -u main_bertpho.py --PATH_STAR_RATING_TRAIN dataset/aivivn/train.csv --PATH_STAR_RATING_TEST dataset/aivivn/test.csv --OUTPUT_DIR output/aivivn/models/bertpho/

## tiki
# rnn
python -u main.py -c config/tiki/rnn/1.train.config
python -u main.py -c config/tiki/rnn/2.train.config
python -u main.py -c config/tiki/rnn/3.train.config
python -u main.py -c config/tiki/rnn/4.train.config
# cnn
python -u main.py -c config/tiki/cnn/1.train.config
# rcnn
python -u main.py -c config/tiki/rcnn/1.train.config
# dpcnn
python -u main.py -c config/tiki/dpcnn/1.train.config
# transformer
python -u main.py -c config/tiki/transformer/1.train.config
#bert
python -u main_bert.py --PATH_STAR_RATING_TRAIN dataset/tiki/train.csv --PATH_STAR_RATING_TEST dataset/tiki/test.csv --OUTPUT_DIR output/tiki/models/bert/
#phoBert
python -u main_bertpho.py --PATH_STAR_RATING_TRAIN dataset/tiki/train.csv --PATH_STAR_RATING_TEST dataset/tiki/test.csv --OUTPUT_DIR output/tiki/models/bertpho/
