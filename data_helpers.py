from lxml import etree
import numpy as np


class DataHelpers:
    def load_dev_data(self):
        return self.__load_data_and_labels('datasets/dev.xml')

    def load_train_data(self):
        return self.__load_data_and_labels('datasets/train.xml')

    def __load_data_and_labels(self, path):
        """
            Loads MR polarity data from files, splits the data into words and generates labels.
            Returns split sentences and labels.
        """
        result = {'positive': [], 'negative': [], 'neutral': []}

        documents = tree = etree.parse(path).getroot()
        for document in documents:
            text = document.find('text').text
            sentiment = document.find('sentiment').text
            result[sentiment].append(text)

        self.__print_dict_info(result)

        positive_examples = result['positive']
        negative_examples = result['negative']

        x_text = positive_examples + negative_examples
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]

        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
            Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    @staticmethod
    def __print_dict_info(data):
        print('positive: ' + str(len(data['positive'])))
        print('neutral: ' + str(len(data['neutral'])))
        print('negative: ' + str(len(data['negative'])))
