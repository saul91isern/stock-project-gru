from tensorflow.keras.layers import Activation, Dense, Dropout, Embedding, LSTM, GRU
from tensorflow.keras.models import Sequential

class GRUModel:
    
    def build(self, amount_of_features, seq_len):
            
        model = Sequential()

        model.add(GRU(
            units=256,
            activation="tanh",
            return_sequences=True,
            dropout=0.2,
            input_shape=(seq_len, amount_of_features)
            )
        )

        model.add(GRU(
            units=256,
            activation="tanh",
            return_sequences=False,
            dropout=0.2
            )
        )

        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="linear"))

        model.summary()

        model.compile(
            optimizer="Nadam",
            loss="mse"
        )

        return model

    def train_model(
        self,
        model, 
        x_train, 
        y_train,
        batch_size,
        epochs, 
        path
    ):
    
        history = model.fit(
            x=x_train, 
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            shuffle=False
        )
        # Once finished the training the model it will be saved and we will delete
        # the class attribute from memory.
        model.save(path)
        del model

        return history