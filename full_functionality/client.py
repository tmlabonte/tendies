from glob import glob
import base64
import requests
import json
import argparse
import sys
import os


class Client:
    """ Client for a TensorFlow ModelServer.

        Performs inference on a directory of images by sending them
        to a TensorFlow-Serving ModelServer, using its RESTful API.

        This version requires input and output to be images; see
        ObjectDetectionClient.py for a TF Object Detection API compliant class.
    """

    def __init__(self,
                 url,
                 input_dir,
                 input_extension,
                 output_dir,
                 output_filename,
                 encoding):
        """ Initializes a Client object.

            Args:
                url: The URL of the TensorFlow ModelServer.
                input_dir: The name of the input directory.
                input_extension: The file extension of input files.
                output_dir: The name of the output directory.
                output_filename: The filename (less extension) of output files.
                encoding: The type of string encoding to use.
        """

        self.url = url
        self.input_dir = input_dir
        self.input_extension = input_extension
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.encoding = encoding

    def visualize(self, input_image, response, i):
        """ Decodes Base64 response data and saves images to a directory.

            Args:
                input_image: The string representing the input image. Not used
                    in this method, but required for child classes to overload.
                response: The list of response dictionaries from the server.
                i: An integer used in iteration over input images.
        """

        # Interprets bitstring output
        response_string = response["b64"]
        encoded_response_string = response_string.encode(self.encoding)
        response_image = base64.b64decode(encoded_response_string)

        # Saves inferred image
        output_file = self.output_dir + "/images/"
        output_file += self.output_filename + str(i) + ".png"
        with open(output_file, "wb") as output_file:
            output_file.write(response_image)

    def inference(self):
        """ Performs inference on a directory of images by sending them
            to the TensorFlow-Serving ModelServer, using its RESTful API.

            Outputs JSON data into txt files and inferred images as png files.
        """

        # Creates output directories
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.output_dir + "/text"):
            os.mkdir(self.output_dir + "/text")
        if not os.path.exists(self.output_dir + "/images"):
            os.mkdir(self.output_dir + "/images")

        # Sends images from input directory
        input_glob = glob(self.input_dir + "/*" + self.input_extension)
        for i, img in enumerate(input_glob):
            # Encodes image in b64
            input_image = open(img, "rb").read()
            input64 = base64.b64encode(input_image)
            input_string = input64.decode(self.encoding)

            # Wraps bitstring in JSON and POSTs, then waits for response
            instance = [{"b64": input_string}]
            data = json.dumps({"instances": instance})
            json_response = requests.post(self.url, data=data)

            # Write output to a .txt file
            output_file = self.output_dir + "/text/output" + str(i) + ".txt"
            with open(output_file, "w") as out:
                out.write(json_response.text)

            # Extracts text from JSON
            response = json.loads(json_response.text)
            response = response["predictions"][0]

            # Visualizes inferred image
            self.visualize(input_image, response, i)


def example_usage(_):
    # Instantiates a Client
    client = Client(FLAGS.url,
                    FLAGS.input_dir,
                    FLAGS.input_extension,
                    FLAGS.output_dir,
                    FLAGS.output_filename,
                    FLAGS.encoding)
    # Performs inference
    client.inference()


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
                        default="output",
                        help="Output file name")

    parser.add_argument("--encoding",
                        type=str,
                        default="utf-8",
                        help="Encoding type")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the inference
    example_usage([sys.argv[0]] + unparsed)
