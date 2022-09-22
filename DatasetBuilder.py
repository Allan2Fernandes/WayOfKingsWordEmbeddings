import os
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical


class DatasetBuilder:
    def __init__(self, root_path):
        directories = os.listdir(root_path)
        self.chapter_paths = []

        for directory in directories:
            self.chapter_paths.append(os.path.join(root_path, directory))
            pass

        pass

    def read_file(self, path):
        with open(path, encoding='utf8') as text_file:
            sentences = text_file.readlines()
            pass
        return sentences

    def read_all_sentences_from_all_chapters(self):
        sentences_of_each_chapter = []
        sentences = []

        for chapter_path in self.chapter_paths:
            sentences_of_each_chapter.append(self.read_file(chapter_path))
            pass

        for each_item in sentences_of_each_chapter:
            for each_sentence in each_item:
                if len(each_sentence) <= 1:
                    continue
                #print(each_sentence)
                sentences.append(each_sentence)
                pass
            pass
        return sentences

    def build_padded_sequences(self):
        my_tokenizer = Tokenizer(oov_token="oov_token", filters='')
        sentences = self.read_all_sentences_from_all_chapters()
        my_tokenizer.fit_on_texts(sentences)
        #print(my_tokenizer.word_index.get("."))
        sequences = my_tokenizer.texts_to_sequences(sentences)
        #padded_sequences = pad_sequences(sequences, padding='pre')
        return sequences, my_tokenizer

    def build_datsets(self, validation_split):
        #Now build training input, training label, validation input and validation label datasets

        sequences, my_tokenizer = self.build_padded_sequences()
        input_sequences = []
        label_sequences = []

        for j in range(len(sequences)):
            one_sequence = sequences[j]
            for i in range(2, len(one_sequence)):
                #print(one_sequence[0:i]) #THe input sequence
                #print("Label is " + str(one_sequence[i])) #The label
                input_sequences.append(one_sequence[0:i])
                label_sequences.append(one_sequence[i])

                pass
            pass

        padded_sequences = pad_sequences(input_sequences, padding = 'pre')


        one_hot_labels = to_categorical(label_sequences)
        #print(one_hot_labels)
        #print(padded_sequences)
        #print(padded_sequences.shape)
        #print(one_hot_labels.shape)
        #print(my_tokenizer.word_index)

        print(len(my_tokenizer.word_index))
        print(my_tokenizer.word_index)

        num_of_words = len(my_tokenizer.word_index) + 1

        return padded_sequences, one_hot_labels, num_of_words










