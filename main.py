from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import DatasetBuilder as db
import Recurrent_Neural_Network as rnn


my_dataset_builder = db.DatasetBuilder(root_path="Way of kings")
my_dataset_builder.build_datsets(validation_split=0.2)

my_rnn = rnn.Reccurent_Neural_Network()


"""
with open("Way of kings/Prelude.txt", encoding='utf8') as text_file:
    sentences = text_file.readlines()
    pass

my_tokenizer = Tokenizer(oov_token='oov_token')
my_tokenizer.fit_on_texts(sentences)

sequences = my_tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, padding='pre')
"""