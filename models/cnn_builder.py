from keras.models import Sequential
from keras.layers import (
    Conv2D,
    Dense,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    Flatten,
    Input,
)
from keras.constraints import UnitNorm


class CNNBuilder:

    def __init__(
        self, in_shape, out_shape, convolutional_layers, fully_connected_layers
    ):

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.convolutional_layers = convolutional_layers
        self.fully_connected_layers = fully_connected_layers

        self.apply_regularization = False
        self.apply_dropout = False
        self.apply_batch_normalization = False
        self.weight_constraints = False

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=self.in_shape))

        for index in range(len(self.convolutional_layers)):
            self.add_convolutional_layer(
                model=model,
                filters=self.convolutional_layers[index],
            )

        model.add(Flatten())

        for index in range(len(self.fully_connected_layers)):
            self.add_fully_connected_layer(
                model=model, nr_neurons=self.fully_connected_layers[index]
            )

        model.add(Dense(self.out_shape, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        return model

    def add_convolutional_layer(self, model, filters):
        model_configuration = {}

        if self.weight_constraints:
            model_configuration["kernel_constraint"] = UnitNorm()

        if self.apply_regularization:
            model_configuration["kernel_regularizer"] = "l2"

        model.add(
            Conv2D(
                filters=filters,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                **model_configuration,
            )
        )

        if self.apply_batch_normalization:
            model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

    def add_fully_connected_layer(self, model, nr_neurons):
        model.add(Dense(nr_neurons, activation="relu"))

        if self.apply_batch_normalization:
            model.add(BatchNormalization())

        if self.apply_dropout:
            model.add(Dropout(rate=0.5))
