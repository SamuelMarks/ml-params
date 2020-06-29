from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    Trainer must be implemented for each ML framework
    """

    @abstractmethod
    def load_data(self, dataset_name, data_loader, data_type='tensorflow_datasets', output_type='numpy'):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: str

        :param data_loader: function that returns the expected data type
        :type (*args, **kwargs) -> not None

        :param data_type:

        :param output_type:

        :return:
        """
        assert dataset_name is not None
        assert data_loader is not None

        loaded_data = data_loader(dataset_name)
        if output_type == 'numpy':
            return loaded_data.as_numpy()
        return loaded_data

    @abstractmethod
    def train(self, epochs, *args, **kwargs):
        """

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: int

        :param args:
        :param kwargs:
        :return:
        """
        """
        Run the training loop for your ML pipeline.
        """
        assert epochs is not None and epochs > 0
