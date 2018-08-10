import argparse
import os
import sys

import keras.backend.tensorflow_backend as K
import tensorflow as tf

from tensorflow.saved_model.builder import SavedModelBuilder
from tensorflow.saved_model.signature_def_utils import build_signature_def
from tensorflow.saved_model.signature_constants \
    import DEFAULT_SERVING_SIGNATURE_DEF_KEY
from tensorflow.saved_model.signature_constants import PREDICT_METHOD_NAME
from tensorflow.saved_model.tag_constants import SERVING
from tensorflow.saved_model.utils import build_tensor_info

from LayerInjector import LayerInjector


class ServerBuilder:
    """ Exports TensorFlow image-based model for serving with RESTful API.

        For TensorFlow-only models, please use the build_server_for_tf
        function. For Keras models, use build_server_for_keras. Please view
        the provided example cases in example_usage below.

        ServerBuilder methodology:
        1. Injects selected input and output layers to the model.
        2. Converts model checkpoint to ProtoBuf (TensorFlow version only).
        3. Wraps in a SavedModel with a PREDICT signature definition.
    """

    def __init__(self):
        pass

    def _create_savedmodel(self,
                           save_path,
                           input_tensor_info_dict,
                           output_tensor_info_dict,
                           graph=None):
        """ Creates a SavedModel by building signature via tensor info,
            then writes to disk.

            Args:
                save_path: The save path for the SavedModel.
                input_tensor_info_dict: A dictionary containing the input
                    tensor info for the SavedModel.
                output_tensor_info_dict: A dictionary containing the output
                    tensor info for the SavedModel.
                graph: The graph to run the tf.Session with. Defaults to
                    tf.Graph().
        """

        # Graph is default if unspecified, otherwise set it to the argument
        graph = tf.Graph() if graph is None else graph

        # Instantiates a SavedModelBuilder
        builder = SavedModelBuilder(save_path)

        # Creates signature for prediction
        signature_definition = build_signature_def(
            input_tensor_info_dict,
            output_tensor_info_dict,
            PREDICT_METHOD_NAME)

        with tf.Session(graph=graph) as sess:
            # Initializes model and variables
            sess.run(tf.global_variables_initializer())

            # Adds meta-information
            builder.add_meta_graph_and_variables(
                sess, [SERVING],
                signature_def_map={
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
                })

        # Writes the SavedModel to disk
        builder.save()

    def _export_graph_to_protobuf(self,
                                  inference_function,
                                  preprocess_function,
                                  postprocess_function,
                                  model_name,
                                  model_version,
                                  checkpoint_dir,
                                  channels,
                                  optional_preprocess_args,
                                  optional_postprocess_args):
        """ Injects input and output layers, then
            exports model graph to ProtoBuf.

            Args:
                inference_function: A function from a TensorFlow model
                    which performs an inference.
                preprocess_function: A function from the LayerInjector class
                    which preprocesses input.
                postprocess_function: A function from the LayerInjector class
                    which postprocesses output.
                model_name: The name of the model.
                model_version: The version number of the model.
                checkpoint_dir: The path to the model's checkpoints directory.
                channels: The number of channels of the input image.
                optional_preprocess_args: Optional list of arguments for use
                    with custom preprocess functions.
                optional_postprocess_args: Optional list of arguments for use
                    with custom postprocess functions.

            Returns:
                output_node_names: A list of the graph's output nodes.
                output_as_image: A boolean set to True if the output is an
                    encoded image.
        """

        # Injects a bitstring layer into beginning of model
        input_bytes = tf.placeholder(tf.string,
                                     shape=[],
                                     name="input_bytes")

        # Sets graph
        graph = input_bytes.graph

        # Preprocesses input bitstring
        input_tensor = preprocess_function(input_bytes,
                                           channels,
                                           optional_preprocess_args)

        # Gets output tensor(s)
        inference_output = inference_function(input_tensor)

        # Postprocesses output tensor(s)
        output_node_names, output_as_image = postprocess_function(
            inference_output,
            optional_postprocess_args)

        # Instantiates a Saver
        saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            # Initializes model and variables
            sess.run(tf.global_variables_initializer())

            # Accesses variables and weights from last checkpoint
            latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, latest_ckpt)

            # Exports graph to ProtoBuf
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), output_node_names)

            # Saves ProtoBuf to disk
            tf.train.write_graph(output_graph_def,
                                 "./",
                                 "temp.pb",
                                 as_text=False)

        # Returns output node names and image boolean
        return output_node_names, output_as_image

    def _wrap_savedmodel_around_protobuf(self,
                                         output_node_names,
                                         output_as_image,
                                         model_name,
                                         model_version,
                                         serve_dir):
        """ Wraps a SavedModel around a ProtoBuf file with a PREDICT
            signature definition for using the TensorFlow-Serving RESTful API.

            Designed to be called immediately after
            ServerBuilder._export_graph_to_protobuf(), with that function's
            outputs as inputs.

            Args:
                output_node_names: A list of the output nodes in the graph.
                    Returned by _export_graph_to_protobuf().
                output_as_image: A boolean telling whether the output of the
                    model is an image. Returned by _export_graph_to_protobuf().
                model_name: The name of the model.
                model_version: The version number of the model.
                serve_dir: The path to the model's serve directory.
        """

        # Parses paths
        # Note that the serve directory MUST have a model version subdirectory
        model_version = str(model_version)
        save_path = serve_dir + "/" + model_name + "/" + model_version

        # Reads in ProtoBuf file
        with tf.gfile.GFile("temp.pb", "rb") as protobuf_file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(protobuf_file.read())

        # Deletes ProtoBuf file
        os.remove("temp.pb")

        # Builds list of input and output tensor names
        tensor_names = ["input_bytes:0"]
        for name in output_node_names:
            tensor_names.append(name + ":0")

        # Gets input and output tensors from GraphDef
        io_tensors = tf.import_graph_def(graph_def,
                                         name="",
                                         return_elements=tensor_names)

        # Separates input and output tensors
        input_bytes = io_tensors[0]
        output_tensors = io_tensors[1:]

        # Builds input prototype
        input_bytes_info = build_tensor_info(input_bytes)
        input_tensor_info = {"input_bytes": input_bytes_info}

        # Builds dictionary of output prototypes
        # Note that output as image MUST have "_bytes" suffix
        if output_as_image:
            # TODO: Implement multi-image outputs
            tensor_info = build_tensor_info(output_tensors[0])
            output_tensor_info = {"output_bytes": tensor_info}
        else:
            output_tensor_info = {}
            for tensor, name in zip(output_tensors, output_node_names):
                tensor_info = build_tensor_info(tensor)
                output_tensor_info[name] = tensor_info

        # Creates and saves SavedModel
        self._create_savedmodel(save_path,
                                input_tensor_info,
                                output_tensor_info,
                                graph=output_tensors[0].graph)

    def _wrap_savedmodel_around_keras(self,
                                      preprocess_function,
                                      postprocess_function,
                                      model_name,
                                      model_version,
                                      h5_filepath,
                                      serve_dir,
                                      channels,
                                      optional_preprocess_args,
                                      optional_postprocess_args):
        """ Injects input and output layers with Keras Lambdas, then
            exports to SavedModel.

            Args:
                preprocess_function: A function from the LayerInjector class
                    which preprocesses input.
                postprocess_function: A function from the LayerInjector class
                    which postprocesses output.
                model_name: The name of the model.
                model_version: The version number of the model.
                h5_filepath: The filepath to a .h5 file containing an
                    exported Keras model.
                serve_dir: The path to the model's serve directory.
                channels: The number of channels of the input image.
                optional_preprocess_args: Optional dict of arguments for use
                    with custom preprocess functions.
                optional_postprocess_args: Optional dict of arguments for use
                    with custom postprocess functions.
        """

        # Parses paths
        # Note that the serve directory MUST have a model version subdirectory
        model_version = str(model_version)
        save_path = serve_dir + "/" + model_name + "/" + model_version

        # Instantiates a Keras model
        K.set_learning_phase(0)
        keras_model = load_model(h5_filepath)

        # Instantiates placeholder for image bitstring
        input_bytes = Input(shape=[], dtype=tf.string)

        # Preprocesses image bitstring
        pre_map = {"channels": channels, **optional_preprocess_args}
        input_tensor = Lambda(preprocess_function,
                              arguments=pre_map)(input_bytes)

        # Gets output tensor(s)
        output_tensor = keras_model(input_tensor)

        # Postprocesses output tensor(s)
        post_map = optional_postprocess_args
        output_bytes = Lambda(postprocess_function,
                              arguments=post_map)(output_tensor)

        # Builds new Model
        model = Model(input_bytes, output_bytes)

        # Builds input/output tensor protos
        input_tensor_info = {"input_bytes": build_tensor_info(model.input)}
        output_tensor_info = {"output_bytes": build_tensor_info(model.output)}

        # Creates and saves SavedModel
        self._create_savedmodel(save_path,
                                input_tensor_info,
                                output_tensor_info)

    def build_server_from_tf(self,
                             inference_function,
                             preprocess_function,
                             postprocess_function,
                             model_name,
                             model_version,
                             checkpoint_dir,
                             serve_dir,
                             channels,
                             optional_preprocess_args=[],
                             optional_postprocess_args=[]):
        """ Builds a Tendies server from a TensorFlow model.

            Args:
                inference_function: A function from a TensorFlow model
                    which performs an inference.
                preprocess_function: A function from the LayerInjector class
                    which preprocesses input.
                postprocess_function: A function from the LayerInjector class
                    which postprocesses output.
                model_name: The name of the model.
                model_version: The version number of the model.
                checkpoint_dir: The path to the model's checkpoints directory.
                serve_dir: The path to the model's serve directory.
                channels: The number of channels of the input image.
                optional_preprocess_args: Optional list of arguments for use
                    with custom preprocess functions.
                optional_postprocess_args: Optional list of arguments for use
                    with custom postprocess functions.
        """

        # Prints export status
        print("Exporting model to ProtoBuf...")

        # Exports given TensorFlow graph to ProtoBuf format
        output_node_names, output_as_image = self._export_graph_to_protobuf(
            inference_function,
            preprocess_function,
            postprocess_function,
            model_name,
            model_version,
            checkpoint_dir,
            channels,
            optional_preprocess_args,
            optional_postprocess_args)

        # Prints export status
        print("Wrapping ProtoBuf in SavedModel...")

        # Wraps a SavedModel around the ProtoBuf
        self._wrap_savedmodel_around_protobuf(
            output_node_names,
            output_as_image,
            model_name,
            model_version,
            serve_dir)

        # Prints export status
        print("Exported successfully!")
        print("""Run the server with:
              tensorflow_model_server --rest_api_port=8501 """
              "--model_name=saved_model --model_base_path=$(path)")

    def build_server_from_keras(self,
                                preprocess_function,
                                postprocess_function,
                                model_name,
                                model_version,
                                h5_filepath,
                                serve_dir,
                                channels,
                                optional_preprocess_args={},
                                optional_postprocess_args={}):
        """ Builds a Tendies server from a Keras model.

            Args:
                preprocess_function: A function from the LayerInjector class
                    which preprocesses input.
                postprocess_function: A function from the LayerInjector class
                    which postprocesses output.
                model_name: The name of the model.
                model_version: The version number of the model.
                h5_filepath: The filepath to a .h5 file containing an
                    exported Keras model.
                serve_dir: The path to the model's serve directory.
                channels: The number of channels of the input image.
                optional_preprocess_args: Optional dict of arguments for use
                    with custom preprocess functions.
                optional_postprocess_args: Optional dict of arguments for use
                    with custom postprocess functions.
        """

        # Prints export status
        print("Exporting Keras model to SavedModel...")

        # Wraps a SavedModel around the given Keras model
        self._wrap_savedmodel_around_keras(
            preprocess_function,
            postprocess_function,
            model_name,
            model_version,
            h5_filepath,
            serve_dir,
            channels,
            optional_preprocess_args,
            optional_postprocess_args)

        # Prints export status
        print("Exported successfully!")
        print("""Run the server with:
              tensorflow_model_server --rest_api_port=8501 """
              "--model_name=saved_model --model_base_path=$(path)")


def example_usage(_):
    # Instantiates a ServerBuilder
    server_builder = ServerBuilder()

    # Instantiates a LayerInjector
    layer_injector = LayerInjector()

    ###################################################################
    # CycleGAN (Image-to-Image in pure TensorFlow)
    ###################################################################
    # sys.path.insert(0, "../CycleGAN-TensorFlow")
    # import model  # nopep8
    # # Instantiates a CycleGAN
    # cycle_gan = model.CycleGAN(ngf=64,
    #                            norm="instance",
    #                            image_size=64)

    # # Builds the server
    # server_builder.build_server_from_tf(
    #     inference_function=cycle_gan.G.sample,
    #     preprocess_function=layer_injector.bitstring_to_float32_tensor,
    #     postprocess_function=layer_injector.float32_tensor_to_bitstring,
    #     model_name=FLAGS.model_name,
    #     model_version=FLAGS.model_version,
    #     checkpoint_dir=FLAGS.checkpoint_dir,
    #     serve_dir=FLAGS.serve_dir,
    #     channels=FLAGS.channels)

    ###################################################################
    # Faster R-CNN (Image to Object Detection API Tensors to Image in pure TF)
    ###################################################################
    # sys.path.insert(0, "C:\\Users\\Tyler Labonte\\Desktop\\models\\research\\object_detection\\builders")  # nopep8
    # sys.path.insert(0, "C:\\Users\\Tyler Labonte\\Desktop\\models\\research\\object_detection\\protos")  # nopep8
    # import model_builder  # nopep8
    # import pipeline_pb2  # nopep8
    # from google.protobuf import text_format  # nopep8
    # CONFIG_FILE_PATH = "C:\\Users\\Tyler Labonte\\Desktop\\rcnn\\pipeline.config"  # nopep8

    # # Builds object detection model from config file
    # pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    # with tf.gfile.GFile(CONFIG_FILE_PATH, 'r') as config:
    #     text_format.Merge(config.read(), pipeline_config)

    # detection_model = model_builder.build(pipeline_config.model,
    #                                       is_training=False)

    # # Creates inference function, encapsulating object detection requirements
    # def object_detection_inference(input_tensors):
    #     inputs = tf.to_float(input_tensors)
    #     preprocessed_inputs, true_image_shapes = detection_model.preprocess(
    #         inputs)
    #     output_tensors = detection_model.predict(
    #         preprocessed_inputs, true_image_shapes)
    #     postprocessed_tensors = detection_model.postprocess(
    #         output_tensors, true_image_shapes)
    #     return postprocessed_tensors

    # # Builds the server
    # server_builder.build_server_from_tf(
    #     inference_function=object_detection_inference,
    #     preprocess_function=layer_injector.bitstring_to_uint8_tensor,
    #     postprocess_function=layer_injector.object_detection_dict_to_tensor_dict,  # nopep8
    #     model_name=FLAGS.model_name,
    #     model_version=FLAGS.model_version,
    #     checkpoint_dir=FLAGS.checkpoint_dir,
    #     serve_dir=FLAGS.serve_dir,
    #     channels=FLAGS.channels)

    ###################################################################
    # Arbitrary Keras Model (Image-to-Image Segmentation in Keras)
    ###################################################################
    # Builds the server
    # server_builder.build_server_from_keras(
    #     preprocess_function=layer_injector.bitstring_to_float32_tensor,
    #     postprocess_function=layer_injector.segmentation_map_to_bitstring_keras,
    #     model_name=FLAGS.model_name,
    #     model_version=FLAGS.model_version,
    #     h5_filepath=FLAGS.h5_filepath,
    #     serve_dir=FLAGS.serve_dir,
    #     channels=FLAGS.channels)


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

    parser.add_argument("--h5_filepath",
                        type=str,
                        default=None,
                        help="Keras model filepath")

    parser.add_argument("--checkpoint_dir",
                        type=str,
                        default=None,
                        help="Path to checkpoints directory")

    parser.add_argument("--serve_dir",
                        type=str,
                        default="serve",
                        help="Path to serve directory")

    parser.add_argument("--channels",
                        type=int,
                        default=3,
                        help="Input image channels")

    # Parses known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Runs the tensorflow app
    tf.app.run(main=example_usage, argv=[sys.argv[0]] + unparsed)
