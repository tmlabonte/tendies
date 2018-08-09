import tensorflow as tf
import math


class LayerInjector:
    """ Contains layer injection functions for ServerBuilder. These layers
        are used for preprocessing and postprocessing.

        Each preprocessing function must take as arguments serving input
        (usually an image bitstring), channels and *args, where *args can be
        used to represent any number of positional arguments. It will return
        the model input.

        Each postprocessing function must take as arguments model output and
        *args, where *args can be used to represent any number of positional
        arguments. It will return the list of output node names and whether the
        output should be transmitted as an image.

        Users of ServerBuilder can utilize *args by passing a list of
        arguments as the optional_preprocess_args or optional_postprocess_args
        parameters in ServerBuilder.build_server().

        Note: Keras functions should use **kwargs instead of *args to be
        compatible with Lambda layers.
    """

    def __init__(self):
        pass

    def bitstring_to_float32_tensor(self,
                                    input_bytes,
                                    channels,
                                    *args):
        """ Transforms image bitstring to float32 tensor.

            Args:
                input_bytes: A bitstring representative of an input image.
                channels: The number of channels in the input image.

            Returns:
                input_tensor: A batched float32 tensor representative of
                    the input image.
        """

        input_bytes = tf.reshape(input_bytes, [])

        # Transforms bitstring to uint8 tensor
        input_tensor = tf.image.decode_png(input_bytes, channels=channels)

        # Converts to float32 tensor
        input_tensor = tf.image.convert_image_dtype(input_tensor, tf.float32)

        # Expands the single tensor into a batch of 1
        input_tensor = tf.expand_dims(input_tensor, 0)
        return input_tensor

    def bitstring_to_uint8_tensor(self,
                                  input_bytes,
                                  channels,
                                  *args):
        """ Transforms image bitstring to uint8 tensor.

            Args:
                input_bytes: A bitstring representative of an input image.
                channels: The number of channels of the input image.

            Returns:
                input_tensor: A batched uint8 tensor representative of
                    the input image.
        """

        input_bytes = tf.reshape(input_bytes, [])

        # Transforms bitstring to uint8 tensor
        input_tensor = tf.image.decode_png(input_bytes, channels=channels)

        # Expands the single tensor into a batch of 1
        input_tensor = tf.expand_dims(input_tensor, 0)
        return input_tensor

    def float32_tensor_to_bitstring(self, output_tensor, *args):
        """ Transforms float32 tensor to bitstring and returns nodes.

            Args:
                output_tensor: A float32 tensor representative of
                    an inferred image.

            Returns:
                output_node_names: A list containing the name of the output
                    node in the graph.
                output_as_image: A boolean telling ServerBuilder that the
                    server output is an encoded image.
        """

        # Sets output to an image
        OUTPUT_AS_IMAGE = True

        # Removes batch dimension
        output_tensor = tf.squeeze(output_tensor)

        # Converts to uint8 tensor
        output_tensor = tf.image.convert_image_dtype(output_tensor, tf.uint8)

        # Transforms uint8 tensor to bitstring
        output_bytes = tf.image.encode_png(output_tensor)

        # Expands the single tensor into a batch of 1
        output_bytes = tf.expand_dims(output_bytes, 0)

        # Adds name to bitstring tensor
        output_bytes = tf.identity(output_bytes, name="output_bytes")

        # Adds output node name to list
        output_node_names = ["output_bytes"]

        # Returns output list and image boolean
        return output_node_names, OUTPUT_AS_IMAGE

    def object_detection_dict_to_tensor_dict(self,
                                             object_detection_tensor_dict,
                                             *args):
        """ Transforms output dict from TensorFlow Object Detection API-
            compliant model to a ServerBuilder-expected dict.

            Args:
                object_detection_tensor_dict: An output dict from a TensorFlow
                    Object Detection API-compliant model. Contains the keys:
                    -"num_detections"
                    -"detection_boxes"
                    -"detection_scores"
                    -"detection_classes"
                    -"detection_keypoints"
                    -"detection_masks"

            Returns:
                output_node_names: A list containing the name of the output
                    nodes in the graph.
                output_as_image: A boolean telling ServerBuilder that the
                    server output is not an encoded image.
        """

        # Sets output to a non-image
        OUTPUT_AS_IMAGE = False
        # Class labels are 1-indexed
        LABEL_ID_OFFSET = 1

        # Assigns names to tensors and adds them to output list
        output_node_names = []
        for name, tensor in object_detection_tensor_dict.items():
            if name == "detection_classes":
                tensor += LABEL_ID_OFFSET
            tensor = tf.identity(tensor, name)
            output_node_names.append(name)

        # Returns output list and image boolean
        return output_node_names, OUTPUT_AS_IMAGE

    def float32_tensor_to_bitstring_keras(self, output_tensor, **kwargs):
        """ Transforms float32 tensor to bitstring tensor.

            Args:
                output_tensor: A float32 tensor representative of
                    an inferred image.

            Returns:
                output_bytes: A bitstring tensor representative of
                    an inferred image.
        """

        # Removes batch dimension
        output_tensor = tf.squeeze(output_tensor)

        # Converts to uint8 tensor
        output_tensor = tf.image.convert_image_dtype(output_tensor, tf.uint8)

        # Transforms uint8 tensor to bitstring
        output_bytes = tf.image.encode_png(output_tensor)

        # Expands the single tensor into a batch of 1
        output_bytes = tf.expand_dims(output_bytes, 0)

        # Adds name to bitstring tensor
        output_bytes = tf.identity(output_bytes, name="output_bytes")

        # Returns bitstring tensor
        return output_bytes

    def segmentation_map_to_bitstring_keras(self, output_tensor, **kwargs):
        """ Transforms segmentation map to bitstring tensor.

            Args:
                output_tensor: A float32 tensor representative of
                    an segmentation map.

            Returns:
                output_bytes: A bitstring tensor representative of
                    an inferred segmentation map.
        """

        # Removes batch dimension
        output_tensor = tf.squeeze(output_tensor)

        # Reshapes tensor from [x^2, n] to [x, x, n]
        # x is the length of a side, n is the number of segmentation classes
        # x = output_tensor.get_shape().as_list()[0]
        # n = output_tensor.get_shape().as_list()[1]
        # output_tensor = tf.reshape(output_tensor,
        #                            [int(math.sqrt(x)), int(math.sqrt(x)), n])

        # Sets classes by choosing the highest, reducing channels to 1
        output_tensor = tf.argmax(output_tensor, axis=2)
        output_tensor = tf.expand_dims(output_tensor, 2)

        # Converts to uint8 tensor
        output_tensor = tf.cast(output_tensor, tf.uint8)
        # output_tensor *= 255

        # Transforms uint8 tensor to bitstring
        output_bytes = tf.image.encode_png(output_tensor)

        # Expands the single tensor into a batch of 1
        output_bytes = tf.expand_dims(output_bytes, 0)

        # Adds name to bitstring tensor
        output_bytes = tf.identity(output_bytes, name="output_bytes")

        # Returns bitstring tensor
        return output_bytes
