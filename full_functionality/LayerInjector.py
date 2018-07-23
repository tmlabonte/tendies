import tensorflow as tf


class LayerInjector:
    def __init__(self):
        pass

    def bitstring_to_float32_tensor(self, input_bytes, image_size, *args):
        """ Transforms image bitstring to float32 tensor.

            Args:
                input_bytes: A bitstring representative of an input image.
                image_size: The input image size (e.g., 64).

            Returns:
                A float32 tensor representative of the input image.
        """
        input_bytes = tf.reshape(input_bytes, [])

        # Transforms bitstring to uint8 tensor
        input_tensor = tf.image.decode_png(input_bytes, channels=3)
        input_tensor = tf.Print(input_tensor, [input_tensor])

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
        input_bytes = tf.reshape(input_bytes, [])

        input_tensor = tf.image.decode_png(input_bytes, channels=3)

        input_tensor = tf.reshape(input_tensor, [image_size, image_size, 3])

        input_tensor = tf.expand_dims(input_tensor, 0)
        return input_tensor

    def float32_tensor_to_bitstring(self, output_tensor, *args):
        """ Transforms float32 tensor to image bitstring.

            Args:
                output_tensor: A float32 tensor representative of
                    an inferred image.

            Returns:
                A bitstring representative of the inferred image.
        """

        # Converts to uint8 tensor
        output_tensor = (output_tensor + 1.0) / 2.0
        output_tensor = tf.image.convert_image_dtype(output_tensor, tf.uint8)

        # Removes the batch dimension
        output_tensor = tf.squeeze(output_tensor, [0])

        # Transforms uint8 tensor to bitstring
        output_bytes = tf.image.encode_png(output_tensor)

        output_dict = {}
        output_dict["output_bytes"] = tf.identity(output_bytes, name="output_bytes")

        return output_dict, True

    def object_detection_dict_to_tensor_dict(
                                        self,
                                        object_detection_tensor_dict,
                                        *args):
        # Class labels are 1-indexed
        label_id_offset = 1

        boxes = object_detection_tensor_dict.get("detection_boxes")
        scores = object_detection_tensor_dict.get("detection_scores")
        classes = object_detection_tensor_dict.get(
          "detection_classes") + label_id_offset
        # keypoints = object_detection_tensor_dict.get("detection_keypoints")
        # masks = object_detection_tensor_dict.get("detection_masks")
        num_detections = object_detection_tensor_dict.get("num_detections")

        output_dict = {}
        output_dict["detection_boxes"] = tf.identity(
          boxes, name="detection_boxes")
        output_dict["detection_scores"] = tf.identity(
          scores, name="detection_scores")
        output_dict["detection_classes"] = tf.identity(
          classes, name="detection_classes")
        output_dict["num_detections"] = tf.identity(
          num_detections, name="num_detections")

        tensor_info_dict = {}
        for k, v in output_dict.items():
            tensor_info_dict[k] = tf.saved_model.utils.build_tensor_info(v)
        print(tensor_info_dict)

        return output_dict, False

        # if keypoints is not None:
        # outputs["detection_keypoints] = tf.identity(
        #     keypoints, name="detection_keypoints)
        # if masks is not None:
        # outputs["detection_masks] = tf.identity(
        #     masks, name="detection_masks)
