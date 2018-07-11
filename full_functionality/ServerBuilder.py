import tensorflow as tf
import argparse
import sys


class ServerBuilder:
    """ Exports TensorFlow model for serving with RESTful API.

        1. Injects bitstring input and output layers to the model.
        2. Converts model checkpoint to ProtoBuf.
        3. Wraps in a SavedModel with a PREDICT signature definition.

        Requires that the input and output of the model's inference
        function is a float32 tensor.
    """

    def __init__(self):
        pass

    def preprocess_bitstring_to_float_tensor(input_bytes, image_size):
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

        # Converts to float32 tensor
        input_tensor = tf.image.convert_image_dtype(input_tensor,
                                                    dtype=tf.float32)
        input_tensor = input_tensor / 127.5 - 1.0

        # Ensures tensor has correct shape
        input_tensor = tf.reshape(input_tensor, [image_size, image_size, 3])

        # Expands the single tensor into a batch of 1
        input_tensor = tf.expand_dims(input_tensor, 0)
        return input_tensor

    def postprocess_float_tensor_to_bitstring(output_tensor):
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
        output_bytes = tf.identity(output_bytes, name="output_bytes")
        return output_bytes

    def export_graph(self,
                     model_instance_inference_function,
                     model_name,
                     model_version,
                     checkpoint_dir,
                     protobuf_dir,
                     image_size):
        """ Injects bitstring input and output layers, then
            exports model graph to ProtoBuf.

            Args:
                model_instance_inference_function: A function which accepts
                    a float32 tensor, performs inference, and returns the
                    resultant float32 tensor.
                model_name: The name of the model.
                model_version: The version number of the model.
                checkpoint_dir: The path to the model's checkpoints directory.
                protobuf_dir: The path to the model's protobuf directory.
                image_size: The input image size (e.g., 64).
        """

        graph = tf.Graph()

        with graph.as_default():
            # Creates placeholder for input bitstring
            # Injects a bitstring layer into beginning of model
            input_bytes = tf.placeholder(tf.string,
                                         shape=[],
                                         name="input_bytes")

            # Preprocesses input bitstring
            input_tensor = self.__preprocess_bitstring_to_float_tensor(
                input_bytes, image_size)

            # Gets output tensor
            output_tensor = model_instance_inference_function(input_tensor)

            # Postprocesses output tensor
            output_bytes = self.__postprocess_float_tensor_to_bitstring(
                output_tensor)

            # Instantiates a Saver
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Accesses variables and weights from last checkpoint
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, latest_ckpt)

            # Exports graph to ProtoBuf
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), [output_bytes.op.name])

            tf.train.write_graph(output_graph_def,
                                 protobuf_dir,
                                 model_name + "_v" + str(model_version),
                                 as_text=False)

    def build_saved_model(self,
                          model_name,
                          model_version,
                          protobuf_dir,
                          serve_dir):
        """ Wraps a SavedModel around a ProtoBuf file with a PREDICT
            signature definition for using the TensorFlow-Serving RESTful API.

            Args:
                model_name: The name of the model.
                model_version: The version number of the model.
                protobuf_dir: The path to the model's protobuf directory.
                serve_dir: The path to the model's serve directory.
        """

        # Instantiates a SavedModelBuilder
        # Note that the serve directory MUST have a model version subdirectory
        builder = tf.saved_model.builder.SavedModelBuilder(serve_dir +
                                                           "/" +
                                                           str(model_version))

        # Reads in ProtoBuf file
        with tf.gfile.GFile(protobuf_dir +
                            "/" +
                            model_name +
                            "_v" +
                            str(model_version),
                            "rb") as protobuf_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(protobuf_file.read())

        # Gets input and output tensors from GraphDef
        # These are our injected bitstring layers
        [inp, out] = tf.import_graph_def(graph_def,
                                         name="",
                                         return_elements=["input_bytes:0",
                                                          "output_bytes:0"])

        with tf.Session(graph=out.graph) as sess:
            # Signature_definition expects a batch
            # So we'll turn the output bitstring into a batch of 1 element
            out = tf.expand_dims(out, 0)

            # Builds prototypes of input and output
            input_bytes = tf.saved_model.utils.build_tensor_info(inp)
            output_bytes = tf.saved_model.utils.build_tensor_info(out)

            # Creates signature for prediction
            signature_definition = tf.saved_model.signature_def_utils.build_signature_def(  # nopep8
                inputs={"input_bytes": input_bytes},
                outputs={"output_bytes": output_bytes},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)  # nopep8

            # Adds meta-information
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
                })

        # Creates the SavedModel
        builder.save()


def example_usage(_):
    sys.path.insert(0, "../CycleGAN-TensorFlow")
    import model  # nopep8

    # Instantiates a CycleGAN
    cycle_gan = model.CycleGAN(ngf=64,
                               norm="instance",
                               image_size=FLAGS.image_size)

    # Instantiates a ServerBuilder
    server_builder = ServerBuilder()

    # Exports model
    print("Exporting model to ProtoBuf...")
    server_builder.export_graph(cycle_gan.G.sample,
                                FLAGS.model_name,
                                FLAGS.model_version,
                                FLAGS.checkpoint_dir,
                                FLAGS.protobuf_dir,
                                FLAGS.image_size)
    print("Wrapping ProtoBuf in SavedModel...")
    server_builder.build_saved_model(FLAGS.model_name,
                                     FLAGS.model_version,
                                     FLAGS.protobuf_dir,
                                     FLAGS.serve_dir)
    print("Exported successfully!")
    print("""Run the server with:
          tensorflow_model_server --rest_api_port=8501 """
          "--model_name=saved_model --model_base_path=$(path)")


if __name__ == "__main__":
    # Instantiates an arg parser
    parser = argparse.ArgumentParser()

    # Establishes default arguments
    parser.add_argument("--model_name",
                        type=str,
                        default="model",
                        help="Model name")

    parser.add_argument("--model_version",
                        type=int,
                        default=1,
                        help="Model version number")

    parser.add_argument("--checkpoint_dir",
                        type=str,
                        default="",
                        help="Path to checkpoints directory")

    parser.add_argument("--protobuf_dir",
                        type=str,
                        default="",
                        help="Path to protobufs directory")

    parser.add_argument("--serve_dir",
                        type=str,
                        default="serve",
                        help="Path to serve directory")

    parser.add_argument("--image_size",
                        type=int,
                        default=64,
                        help="Image size")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    tf.app.run(main=example_usage, argv=[sys.argv[0]] + unparsed)
