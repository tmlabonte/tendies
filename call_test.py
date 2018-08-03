import tensorflow as tf
from keras.engine.input_layer import Input
from keras.models import Model, load_model
from keras.layers import Lambda
import keras.backend.tensorflow_backend as K
import base64


def bitstring_to_float32_tensor(input_bytes):
    """ Transforms image bitstring to float32 tensor.

        Args:
            input_bytes: A bitstring representative of an input image.

        Returns:
            A batched float32 tensor representative of the input image.
    """

    input_bytes = tf.reshape(input_bytes, [])
    input_bytes = tf.cast(input_bytes, tf.string)

    # Transforms bitstring to uint8 tensor
    input_tensor = tf.image.decode_png(input_bytes, channels=1)

    # Converts to float32 tensor
    input_tensor = tf.image.convert_image_dtype(input_tensor, tf.float32)

    # Ensures tensor has correct shape
    input_tensor = tf.reshape(input_tensor, [28, 28, 1])

    # Expands the single tensor into a batch of 1
    input_tensor = tf.expand_dims(input_tensor, 0)
    return input_tensor


def float32_tensor_to_bitstring(output_tensor):
    """ Transforms float32 tensor to list of image bitstrings.

        Args:
            output_tensor: A float32 tensor representative of
                an inferred image.

        Returns:
            output_node_names: A list containing the name of the output
                node in the graph.
    """

    output_tensor = tf.squeeze(output_tensor)
    output_tensor = tf.argmax(output_tensor, axis=2)
    output_tensor = tf.expand_dims(output_tensor, 2)

    # Converts to uint8 tensor
    output_tensor = tf.image.convert_image_dtype(output_tensor, tf.uint8)

    # Transforms uint8 tensor to bitstring
    output_bytes = tf.image.encode_png(output_tensor)

    output_bytes = tf.identity(output_bytes, name="output_bytes")

    # Expands the single tensor into a batch of 1
    output_bytes = tf.expand_dims(output_bytes, 0)

    # Returns output tensor
    return output_bytes


# Sets model phase to inference
K.set_learning_phase(0)

# Loads model from hdf5 file
sat_net = load_model("mnist_seg.h5")

# Instantiates placeholder for image bitstring
input_bytes = Input(shape=[], dtype=tf.string)

# Converts image bitstring to float32 tensor
input_tensor = Lambda(bitstring_to_float32_tensor)(input_bytes)

# Performs inference on tensor, returning a float32 tensor
output_tensor = sat_net(input_tensor)

# Converts float32 tensor to image bitstring
output_bytes = Lambda(float32_tensor_to_bitstring)(output_tensor)

# Builds new Model
sat_net = Model(input_bytes, output_bytes)
sat_net.summary()

# Creates signature for prediction
signature_definition = tf.saved_model.signature_def_utils.predict_signature_def(
    {"input_bytes": sat_net.input},
    {"output_bytes": sat_net.output})

# Instantiates a SavedModelBuilder
builder = tf.saved_model.builder.SavedModelBuilder("serve/1")

with tf.Session() as sess:
    # Initializes model and variables
    sess.run(tf.global_variables_initializer())

    # Adds meta-information
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
        })

    # Saves the model
    builder.save()
