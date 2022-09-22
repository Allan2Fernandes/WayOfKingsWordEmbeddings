import tensorflow as tf
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

class Reccurent_Neural_Network:
    def __init__(self, max_sequence_length, word_dimensions, vocabulary_size):
        self.max_sequence_length = max_sequence_length
        self.word_dimensions = word_dimensions
        self.vocabulary_size = vocabulary_size
        pass

    def create_model(self):
        embedding_layer = Embedding(input_dim=self.vocabulary_size, output_dim=self.word_dimensions, input_length=self.max_sequence_length-1) #Vectorize the words with 100 dimensions?
        X = Bidirectional(LSTM(units = 512, activation='relu'))(embedding_layer)
        X = Bidirectional(LSTM(units=256, activation='relu'))(X)
        dense_layer = Dense(units=self.vocabulary_size, activation='softmax')(X)

        self.my_model = Model(inputs = embedding_layer, outputs = dense_layer)
        pass

    def compile_model(self):
        self.my_model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=0.0001), metrics = ['accuracy'])

    def summarize_mode(self):
        self.my_model.summary()
        pass