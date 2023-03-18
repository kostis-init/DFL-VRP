import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError


class VRPDataset(tf.keras.utils.Sequence):
    def __init__(self, vrp_instances, batch_size=32):
        self.vrp_instances = vrp_instances
        self.batch_size = batch_size
        self.indices = []
        for i, instance in enumerate(self.vrp_instances):
            for j in range(len(instance.edges)):
                self.indices.append((i, j))

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        features = []
        targets = []
        for i, j in batch_indices:
            instance = self.vrp_instances[i]
            edge = instance.edges[j]
            features.append(edge.features)
            targets.append(edge.cost)
        return tf.convert_to_tensor(features), tf.convert_to_tensor(targets)


class EdgeCostPredictor(Sequential):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.add(Dense(hidden_size, activation='relu', input_shape=(input_size,)))
        self.add(Dense(hidden_size, activation='relu'))
        self.add(Dense(output_size))


class EdgeTrainer:
    def __init__(self, train_set, test_set):
        self.train_dataloader = VRPDataset(train_set)
        self.test_dataloader = VRPDataset(test_set)
        self.model = EdgeCostPredictor(len(train_set[0].edges[0].features), 32, 1)
        self.optimizer = Adam(learning_rate=0.001)
        self.loss_fn = MeanSquaredError()

    def train(self, num_epochs=10):
        # fit model
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.model.fit(self.train_dataloader, epochs=num_epochs)

    def test(self):
        # evaluate the model
        loss = self.model.evaluate(self.test_dataloader, verbose=0)
        print(f'Loss: {loss}')

    def predict(self, features):
        return self.model.predict(features)
