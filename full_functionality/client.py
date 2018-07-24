from glob import glob
import base64
import requests
import json
import argparse
import sys
import os
sys.path.insert(0, "C:\\Users\\Tyler Labonte\\Desktop\\models\\research\\object_detection\\utils")
import visualization_utils  # nopep8
import label_map_util  # nopep8
import tensorflow as tf


def get_category_index(path_to_labels_map):
        label_map = label_map_util.load_labelmap(path_to_labels_map)
        categories = label_map_util.convert_label_map_to_categories(
                        label_map,
                        max_num_classes=1,
                        use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index


def preprocess_bitstring_to_uint8_tensor(input_bytes, image_size):
    input_bytes = tf.reshape(input_bytes, [])

    input_tensor = tf.image.decode_jpeg(input_bytes, channels=3)

    input_tensor = tf.reshape(input_tensor, [image_size, image_size, 3])

    return input_tensor


def inference(_):
    """ Performs inference on a directory of images by sending them
        to the TensorFlow-Serving ModelServer, using its RESTful API.
    """

    # Creates output directory
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)
    if not os.path.exists(FLAGS.output_dir + "/text"):
        os.mkdir(FLAGS.output_dir + "/text")
    if not os.path.exists(FLAGS.output_dir + "/images"):
        os.mkdir(FLAGS.output_dir + "/images")

    # Sends images from input directory
    input_glob = glob(FLAGS.input_dir + "/*" + FLAGS.input_extension)
    for i, img in enumerate(input_glob):
        # Encodes image in b64
        input_image = open(img, "rb").read()
        input64 = base64.b64encode(input_image)
        input_string = input64.decode(FLAGS.encoding)

        # Wraps bitstring in JSON and POST to server, then waits for response
        instance = [{"b64": input_string}]
        data = json.dumps({"instances": instance})
        json_response = requests.post(FLAGS.url, data=data)

        # Write output to a .txt file
        output_file = FLAGS.output_dir + "/text/output" + str(i) + ".txt"
        with open(output_file, "w") as out:
            out.write(json_response.text)

        # Extracts text from JSON
        response = json.loads(json_response.text)
        response = response["predictions"][0]

        # Visualize images
        image = preprocess_bitstring_to_uint8_tensor(input_image, 512)
        detection_boxes = response["detection_boxes"]
        detection_classes = response["detection_classes"]
        detection_scores = response["detection_scores"]

        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            tf.convert_to_tensor(detection_boxes),
            tf.convert_to_tensor(detection_classes),
            scores=tf.convert_to_tensor(detection_scores),
            category_index=get_category_index(FLAGS.label),
            instance_masks=None,
            use_normalized_coordinates=True,
            line_thickness=2)

        # Save images
        output_file = FLAGS.output_dir + "/images/"
        output_file += FLAGS.output_filename + str(i) + FLAGS.output_extension
        visualization_utils.save_image_array_as_png(image, output_file)

        # # Interprets bitstring output
        # response_string = response[0]["b64"]
        # encoded_response_string = response_string.encode(FLAGS.encoding)
        # response_image = base64.b64decode(encoded_response_string)

        # # Saves inferred image
        # output_file = FLAGS.output_dir + "/images/"
        # output_file += FLAGS.output_filename + str(i) + FLAGS.output_extension
        # with open(output_file, "wb") as output_file:
        #     output_file.write(response_image)


if __name__ == "__main__":
    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Establishes default arguments
    parser.add_argument("--url",
                        type=str,
                        default="http://localhost:8501/v1/models/"
                                "saved_model:predict",
                        help="URL of server")

    parser.add_argument("--input_dir",
                        type=str,
                        default="input",
                        help="Path to input directory")

    parser.add_argument("--input_extension",
                        type=str,
                        default=".jpg",
                        help="Input file extension")

    parser.add_argument("--output_dir",
                        type=str,
                        default="output",
                        help="Path to output directory")

    parser.add_argument("--output_filename",
                        type=str,
                        default="output",
                        help="Output file name")

    parser.add_argument("--output_extension",
                        type=str,
                        default=".png",
                        help="Output file extension")

    parser.add_argument("--encoding",
                        type=str,
                        default="utf-8",
                        help="Encoding type")

    parser.add_argument("--label",
                        type=str,
                        default="C:\\Users\\Tyler Labonte\\Desktop\\sat_net\\label_data\\astronet_label_map_2.pbtxt",
                        help="Labelmap path")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the inference
    inference([sys.argv[0]] + unparsed)
