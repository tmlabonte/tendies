from LayerInjector import LayerInjector
import tensorflow as tf
import argparse
import sys


class ServerBuilder:
    """ Exports TensorFlow model for serving with RESTful API.

        1. Injects selected input and output layers to the model.
        2. Converts model checkpoint to ProtoBuf.
        3. Wraps in a SavedModel with a PREDICT signature definition.
    """

    def __init__(self):
        pass

    def export_graph(self,
                     model_instance_inference_function,
                     preprocess_function,
                     postprocess_function,
                     model_name,
                     model_version,
                     checkpoint_dir,
                     protobuf_dir,
                     image_size,
                     optional_preprocess_args=None,
                     optional_postprocess_args=None):
        """ Injects input and output layers, then
            exports model graph to ProtoBuf.

            Args:
                model_instance_inference_function: A function which performs
                    an inference.
                preprocess_function: A function from the LayerInjector class
                    which preprocesses input.
                postprocess_function: A function from the LayerInjector class
                    which postprocesses output.
                model_name: The name of the model.
                model_version: The version number of the model.
                checkpoint_dir: The path to the model's checkpoints directory.
                protobuf_dir: The path to the model's protobuf directory.
                image_size: The input image size (e.g., 64).
                optional_preprocess_args: Optional arguments for use with
                    custom preprocess functions.
                optional_postprocess_args: Optional arguments for use with
                    custom postprocess functions.

            Returns:
                output_node_names: A list of the graph's output nodes.
                output_as_image: A boolean set to True if the output is an
                    encoded image.
        """

        # Creates placeholder for input bitstring
        # Injects a bitstring layer into beginning of model
        input_bytes = tf.placeholder(tf.string,
                                     shape=[],
                                     name="input_bytes")

        # Sets graph
        graph = input_bytes.graph

        # Preprocesses input bitstring
        input_tensor = preprocess_function(input_bytes,
                                           image_size,
                                           optional_preprocess_args)

        # Gets output tensor(s)
        inference_output = model_instance_inference_function(input_tensor)

        # Postprocesses output tensor(s)
        output_node_names, output_as_image = postprocess_function(
                                                inference_output,
                                                optional_postprocess_args)

        with graph.as_default():
            # Instantiates a Saver
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Accesses variables and weights from last checkpoint
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, latest_ckpt)

            # Exports graph to ProtoBuf
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), output_node_names)

            # Saves ProtoBuf to disk
            tf.train.write_graph(output_graph_def,
                                 protobuf_dir,
                                 model_name + "v" + str(model_version) + ".pb",
                                 as_text=False)

        # Returns output node names and image boolean
        return output_node_names, output_as_image

    def build_saved_model(self,
                          output_node_names,
                          output_as_image,
                          model_name,
                          model_version,
                          protobuf_dir,
                          serve_dir):
        """ Wraps a SavedModel around a ProtoBuf file with a PREDICT
            signature definition for using the TensorFlow-Serving RESTful API.

            Args:
                output_node_names: A list of the output nodes in the graph.
                    Returned by export_graph().
                output_as_image: A boolean representing whether the output of
                    the model is an image. Returned by export_graph().
                model_name: The name of the model.
                model_version: The version number of the model.
                protobuf_dir: The path to the model's protobuf directory.
                serve_dir: The path to the model's serve directory.
        """

        # Parses paths
        # Note that the serve directory MUST have a model version subdirectory
        model_version = str(model_version)
        save_path = serve_dir + "/" + model_name + "/" + model_version
        pb_path = protobuf_dir + "/" + model_name + "v" + model_version + ".pb"

        # Instantiates a SavedModelBuilder
        builder = tf.saved_model.builder.SavedModelBuilder(save_path)

        # Reads in ProtoBuf file
        with tf.gfile.GFile(pb_path, "rb") as protobuf_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(protobuf_file.read())

        # Builds list of input and output tensor names
        tensor_names = ["input_bytes:0"]
        for name in output_node_names:
            tensor_names.append(name + ":0")

        # Gets input and output tensors from GraphDef
        io_tensors = tf.import_graph_def(graph_def,
                                         name="",
                                         return_elements=tensor_names)

        # Separates input and output tensors
        input_tensors = []
        output_tensors = []
        for tensor in io_tensors:
            # TODO: shouldn't have to truncate, why does import graph def
            # return tensors whose ops end in _1?
            node_name = tensor.op.name[:-2]
            if node_name in output_node_names:
                output_tensors.append(tensor)
            else:
                input_tensors.append(tensor)

        with tf.Session(graph=output_tensors[0].graph) as sess:
            sess.run(tf.global_variables_initializer())

            # Builds prototype of input
            input_bytes = tf.saved_model.utils.build_tensor_info(
                                                              input_tensors[0])

            # Builds dictionary of output prototypes
            # Note that output as image MUST have "_bytes" suffix
            if output_as_image:
                tensor_info = tf.saved_model.utils.build_tensor_info(
                                                             output_tensors[0])
                output_tensor_info = {"output_bytes": tensor_info}
            else:
                output_tensor_info = {}
                for tensor, name in zip(output_tensors, output_node_names):
                    tensor_info = tf.saved_model.utils.build_tensor_info(
                                                                        tensor)
                    output_tensor_info[name] = tensor_info

            # Creates signature for prediction
            signature_definition = tf.saved_model.signature_def_utils.build_signature_def(  # nopep8
                inputs={"input_bytes": input_bytes},
                outputs=output_tensor_info,
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
    # sys.path.insert(0, "../CycleGAN-TensorFlow")
    sys.path.insert(0, "C:\\Users\\Tyler Labonte\\Desktop\\models\\research\\object_detection\\builders")
    sys.path.insert(0, "C:\\Users\\Tyler Labonte\\Desktop\\models\\research\\object_detection\\protos")
    #import model  # nopep8
    import model_builder  # nopep8
    import pipeline_pb2  # nopep8
    from google.protobuf import text_format  # nopep8

    # # Instantiates a CycleGAN
    # cycle_gan = model.CycleGAN(ngf=64,
    #                            norm="instance",
    #                            image_size=FLAGS.image_size)

    # Builds object detection model from config file
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile("C:\\Users\\Tyler Labonte\\Desktop\\sat_net\\pipeline.config", 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    detection_model = model_builder.build(pipeline_config.model,
                                          is_training=False)

    # Creates inference function, encapsulating object detection requirements
    def object_detection_inference(input_tensors):
        inputs = tf.to_float(input_tensors)
        preprocessed_inputs, true_image_shapes = detection_model.preprocess(
            inputs)
        output_tensors = detection_model.predict(
            preprocessed_inputs, true_image_shapes)
        postprocessed_tensors = detection_model.postprocess(
            output_tensors, true_image_shapes)
        return postprocessed_tensors

    # Instantiates a ServerBuilder
    server_builder = ServerBuilder()

    # Instantiates a LayerInjector
    layer_injector = LayerInjector()

    # Exports model
    print("Exporting model to ProtoBuf...")
    output_node_names, output_as_image = server_builder.export_graph(
                                object_detection_inference,
                                layer_injector.bitstring_to_uint8_tensor,
                                layer_injector.object_detection_dict_to_tensor_dict,
                                FLAGS.model_name,
                                FLAGS.model_version,
                                FLAGS.checkpoint_dir,
                                FLAGS.protobuf_dir,
                                FLAGS.image_size)
    print("Wrapping ProtoBuf in SavedModel...")
    server_builder.build_saved_model(output_node_names,
                                     output_as_image,
                                     FLAGS.model_name,
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
                        default="protobufs",
                        help="Path to protobufs directory")

    parser.add_argument("--serve_dir",
                        type=str,
                        default="serve",
                        help="Path to serve directory")

    parser.add_argument("--image_size",
                        type=int,
                        default=512,
                        help="Image size")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    tf.app.run(main=example_usage, argv=[sys.argv[0]] + unparsed)
