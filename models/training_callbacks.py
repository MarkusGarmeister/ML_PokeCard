from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def default_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
    ]
