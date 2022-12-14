import tensorflow as tf

def main() -> None:

    mnist = tf.keras.datasets.mnist
    ( x_train_int, y_train ), ( x_test_int, y_test ) = mnist.load_data()
    x_train = x_train_int / 255.0
    x_test  = x_test_int / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten( input_shape=( 28, 28 ) ),
        tf.keras.layers.Dense( 128, activation='relu' ),
        tf.keras.layers.Dropout( 0.2 ),
        tf.keras.layers.Dense( 10 )
    ])

    predications = model( x_train[:1] ).numpy()
    print( 'predications = {0}'.format( predications ) )
    print( 'tf.nn.softmax(predications).numpy() = {0}'.format( tf.nn.softmax(predications).numpy() ) )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True )
    loss_fn( y_train[:1], predications ).numpy()
    print( 'loss_fn( y_train[:1], predications ).numpy() = {0}'.format( loss_fn( y_train[:1], predications ).numpy() ) )

    model.compile(
            optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy']
            )
    model.fit( x_train, y_train, epochs=5 )
    model.evaluate( x_test, y_test, verbose=2)

    probability_model = tf.keras.models.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])
    probabilities = probability_model( x_test[:5] ).numpy()
    print( 'probability_model( x_test[:5] ).numpy() = {0}'.format( probabilities ))

if __name__ == '__main__':
    main()
