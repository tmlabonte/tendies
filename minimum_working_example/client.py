import base64
import requests
import json
import argparse


# Send image to server, receive inferred output
def inference():
    # Open and read image
    input_image = open(FLAGS.input_file, "rb").read()

    # Encode image in b64
    encoded_input_string = base64.b64encode(input_image)
    input_string = encoded_input_string.decode(FLAGS.encoding)

    # Wrap bitstring in JSON and POST to server, then wait for response
    instance = [{"b64": input_string}]
    data = json.dumps({"instances": instance})
    json_response = requests.post(FLAGS.url, data=data)

    # Extract text from JSON
    response = json.loads(json_response.text)

    # Interpret bitstring output
    response_string = response["predictions"][0]["b64"]
    encoded_response_string = response_string.encode(FLAGS.encoding)
    response_image = base64.b64decode(encoded_response_string)

    # Save inferred image
    with open(FLAGS.output_file, 'wb') as output_file:
        output_file.write(response_image)


if __name__ == "__main__":
    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Establish default arguments
    parser.add_argument("--url",
                        type=str,
                        default="http://localhost:8501/v1/models/"
                                "saved_model:predict",
                        help="URL of server")

    parser.add_argument("--input_file",
                        type=str,
                        default="../CycleGAN-TensorFlow/data"
                                "/gaussian2sinusoidal64"
                                "/trainA/gaussian0.png",
                        help="Path to input file")

    parser.add_argument("--output_file",
                        type=str,
                        default="sinusoidal.png",
                        help="Path to output file")

    parser.add_argument("--encoding",
                        type=str,
                        default="utf-8",
                        help="Encoding type")

    # Parse known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Run the inference
    inference()
