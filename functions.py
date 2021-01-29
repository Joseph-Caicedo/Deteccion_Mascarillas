import tensorflow as tf
import os


def image_preprocessing(target_size, validation_split, batch_size, data_dir):
    """Preprocesamiento de imágenes .

    Aumento de imagen y estructuración de los conjuntos de entrenamiento y validación:

    Parámetros:
    validation_split -- Porcentaje de división del conjunto de validación

    Devoluciones:
    train_generator -- Conjunto de entrenamiento.
    validation_generator -- Conjunto de validación.

    Excepciones:
    ValueError -- Si (validation_split <= 0).
    ValueError -- Si (target_size <= 0).
    ValueError -- Si (len(target_size) != 2).
    """
    # --------------------
    # Aumento de imagen
    # --------------------
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=validation_split,
                                                                    rescale=1./255,
                                                                    rotation_range=40,
                                                                    width_shift_range=0.2,
                                                                    height_shift_range=0.2,
                                                                    zoom_range=0.2,
                                                                    horizontal_flip=True,
                                                                    fill_mode='nearest')
    # --------------------
    # Imágenes de entrenamiento de flujo en lotes de 20 usando el generador train_datagen
    # --------------------
    train_generator = image_datagen.flow_from_directory(data_dir,
                                                        batch_size=batch_size,
                                                        class_mode='binary',
                                                        target_size=target_size,
                                                        shuffle=True,
                                                        subset='training')
    # --------------------
    # Imágenes de validación de flujo en lotes de 20 usando el generador train_datagen
    # --------------------
    validation_generator = image_datagen.flow_from_directory(data_dir,
                                                             batch_size=batch_size,
                                                             class_mode='binary',
                                                             target_size=target_size,
                                                             shuffle=True,
                                                             subset='validation')
    return train_generator, validation_generator


def sequential_model(input_shape, n, dense_units):
    """Establece la secuencia del modelo.

    Devuelve la secuencia de un modelo basado en el número de capas ingresado:

    model_layers = n*2 + 3

    Parámetros:
    n -- Número de capas de convolución y de agrupación.
    input_shape -- Tamaño de la imagen (datos de entrada).
    dense_units -- Unidades de capa NN densamente conectadas.

    Devoluciones:
    model -- Secuencia del modelo.

    Excepciones:
    ValueError -- Si (n <= 0).
    ValueError -- Si (input_shape <= 0).
    ValueError -- Si (len(input_shape) != 3 or 4).
    """
    model = tf.keras.models.Sequential()
    filters = 16
    for i in range(n):
        # --------------------
        # Convolución
        # --------------------
        model.add(tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D(2, 2))
        filters = filters * 2
    # --------------------
    # Aplanar los resultados para incorporarlos a un DNN
    # --------------------
    model.add(tf.keras.layers.Flatten())
    # --------------------
    # Capa NN regular densamente conectada.
    # --------------------
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    # --------------------
    # Solo 1 neurona de salida. Contendrá un valor de 0-1.
    # --------------------
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


def model_compile(model, learning_rate, optimizer):
    """Compilar modelo.

    Establece la función de perdida y el método de optimización:

    Parámetros:
    model -- Secuencia del modelo.
    learning_rate -- tasa de aprendizaje.
    optimizer -- Entero que determina el optimizador.

    Excepciones:
    ValueError -- Si (learning_rate <= 0).
    """
    if optimizer == 1:
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model


def main():
    data_path = '/home/joseph/Documentos/GitHub/Detencion_Mascarillas/Data_2'
    data_dir = os.path.join(data_path)
    # Hiperparametros
    input_shape = (150, 150, 3)
    validation_split = 0.2
    batch_size = 60
    learning_rate = 0.001

    optimizer = 1       # RMSprop(1) - Adam(2)
    val_accuracy = 0.98
    accuracy = 0.98
    epochs = 3

    layers_numbers = 3
    dense_units = 512

    train_generator, validation_generator = image_preprocessing((input_shape[0], input_shape[1]), validation_split,
                                                                batch_size, data_dir)
    model = sequential_model(input_shape, layers_numbers, dense_units)
    model_compile(model, learning_rate, optimizer)

    class Callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') >= accuracy and logs.get('val_accuracy') >= val_accuracy:
                print("\n¡Alcanzó el {}% de precisión, cancelando el entrenamiento!".format(accuracy*100))
                self.model.stop_training = True

    callbacks = Callback()
    history = model.fit(train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=train_generator.samples // batch_size,
                        epochs=epochs,
                        validation_steps=validation_generator.samples // batch_size,
                        callbacks=[callbacks])

if __name__ == '__main__':
    main()
