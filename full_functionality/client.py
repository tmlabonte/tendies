from glob import glob
import base64
import requests
import json
import argparse
import sys
import os


def inference(_):
    """ Performs inference on a directory of images by sending them
        to the TensorFlow-Serving ModelServer, using its RESTful API.
    """

    # Creates output directory
    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    # Sends images from input directory
    for i, img in enumerate(glob(FLAGS.input_dir +
                                 "/*" +
                                 FLAGS.input_extension)):
        # Encodes image in b64
        input_image = open(img, "rb").read()
        input64 = base64.b64encode(input_image)
        input_string = input64.decode(FLAGS.encoding)

        # Wraps bitstring in JSON and POST to server, then waits for response
        instance = [{"b64": input_string}]
        data = json.dumps({"instances": instance})
        json_response = requests.post(FLAGS.url, data=data)

        # Extracts text from JSON
        print(json_response)
        response = json.loads(json_response.text)

        # Interprets bitstring output
        response_string = response["predictions"][0]["b64"]
        encoded_response_string = response_string.encode(FLAGS.encoding)
        response_image = base64.b64decode(encoded_response_string)

        # Saves inferred image
        output_file = FLAGS.output_dir + "/"
        output_file += FLAGS.output_filename + str(i) + FLAGS.output_extension
        with open(output_file, 'wb') as output_file:
            output_file.write(response_image)


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
                        default=".png",
                        help="Input file extension")

    parser.add_argument("--output_dir",
                        type=str,
                        default="output",
                        help="Path to output directory")

    parser.add_argument("--output_filename",
                        type=str,
                        default="sinusoidal",
                        help="Output file name")

    parser.add_argument("--output_extension",
                        type=str,
                        default=".png",
                        help="Output file extension")

    parser.add_argument("--encoding",
                        type=str,
                        default="utf-8",
                        help="Encoding type")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the inference
    inference([sys.argv[0]] + unparsed)
