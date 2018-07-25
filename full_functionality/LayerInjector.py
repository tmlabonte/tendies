import tensorflow as tf


class LayerInjector:
    """ Contains layer injection functions for ServerBuilder. These layers
        are used for preprocessing and postprocessing.

        Each preprocessing function must take as arguments serving input
        (usually an image bitstring), image_size, and *args, where *args can be
        used to represent any number of positional arguments. It will return
        the model input.

        Each postprocessing function must take as arguments model output and
        *args, where *args can be used to represent any number of positional
        arguments. It will return the list of output node names and whether the
        output should be transmitted as an image.

        Users of ServerBuilder can utilize *args by passing a list of
        arguments as the optional_preprocess_args or optional_postprocess_args
        parameters in ServerBuilder.export_graph().
    """

    def __init__(self):
        pass

    def bitstring_to_float32_tensor(self, input_bytes, image_size, *args):
        """ Transforms image bitstring to float32 tensor.

            Args:
                input_bytes: A bitstring representative of an input image.
                image_size: The input image size (e.g., 64).

            Returns:
                A batched float32 tensor representative of the input image.
        """

        input_bytes = tf.reshape(input_bytes, [])

        # Transforms bitstring to uint8 tensor
        input_tensor = tf.image.decode_png(input_bytes, channels=3)

        # Converts to float32 tensor
        input_tensor = tf.image.convert_image_dtype(input_tensor,
                                                    dtype=tf.float32)
        input_tensor = input_tensor / 127.5 - 1.0

        # Ensures tensor has correct shape
        input_tensor = tf.reshape(input_tensor, [image_size, image_size, 3])

        # Expands the single tensor into a batch of 1
        input_tensor = tf.expand_dims(input_tensor, 0)
        return input_tensor

    def bitstring_to_uint8_tensor(self, input_bytes, image_size, *args):
        """ Transforms image bitstring to uint8 tensor.

            Args:
                input_bytes: A bitstring representative of an input image.
                image_size: The input image size (e.g., 64).

            Returns:
                A batched uint8 tensor representative of the input image.
        """

        input_bytes = tf.reshape(input_bytes, [])

        # Transforms bitstring to uint8 tensor
        input_tensor = tf.image.decode_png(input_bytes, channels=3)

        # Ensures tensor has correct shape
        input_tensor = tf.reshape(input_tensor, [image_size, image_size, 3])

        # Expands the single tensor into a batch of 1
        input_tensor = tf.expand_dims(input_tensor, 0)
        return input_tensor

    def float32_tensor_to_bitstring(self, output_tensor, *args):
        """ Transforms float32 tensor to dict of image bitstring.

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

        # Converts to uint8 tensor
        output_tensor = (output_tensor + 1.0) / 2.0
        output_tensor = tf.image.convert_image_dtype(output_tensor, tf.uint8)

        # Transforms uint8 tensor to bitstring
        output_bytes = tf.image.encode_png(output_tensor)

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
