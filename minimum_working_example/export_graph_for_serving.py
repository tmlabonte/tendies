import tensorflow as tf
import argparse
import sys
sys.path.insert(0, "../CycleGAN-TensorFlow")
import model  # nopep8


# Transform image bitstring to float tensor
def preprocess_bitstring_to_float_tensor(input_bytes, image_size):
    input_bytes = tf.reshape(input_bytes, [])
    input_tensor = tf.image.decode_png(input_bytes, channels=3)
    input_tensor = tf.image.convert_image_dtype(input_tensor,
                                                dtype=tf.float32)
    input_tensor = input_tensor / 127.5 - 1.0
    input_tensor = tf.reshape(input_tensor, [image_size, image_size, 3])
    input_tensor = tf.expand_dims(input_tensor, 0)
    return input_tensor


# Transform float tensor to image bitstring
def postprocess_float_tensor_to_bitstring(output_tensor):
    output_tensor = (output_tensor + 1.0) / 2.0
    output_tensor = tf.image.convert_image_dtype(output_tensor, tf.uint8)
    output_tensor = tf.squeeze(output_tensor, [0])
    output_bytes = tf.image.encode_png(output_tensor)
    output_bytes = tf.identity(output_bytes, name="output_bytes")
    return output_bytes


# Export graph to ProtoBuf
def export_graph():
    graph = tf.Graph()

    with graph.as_default():
        # Instantiate a CycleGAN
        cycle_gan = model.CycleGAN(ngf=64,
                                   norm="instance",
                                   image_size=FLAGS.image_size)

        # Create placeholder for image bitstring
        input_bytes = tf.placeholder(tf.string, shape=[], name="input_bytes")

        # Preprocess input (bitstring to float tensor)
        input_tensor = preprocess_bitstring_to_float_tensor(input_bytes,
                                                            FLAGS.image_size)

        # Get style transferred tensor
        output_tensor = cycle_gan.G.sample(input_tensor)

        # Postprocess output
        output_bytes = postprocess_float_tensor_to_bitstring(output_tensor)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        # Access variables and weights from last checkpoint
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        saver.restore(sess, latest_ckpt)

        # Export graph to ProtoBuf
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_bytes.op.name])

        tf.train.write_graph(output_graph_def,
                             FLAGS.protobuf_dir,
                             FLAGS.model_name + "_v" + str(FLAGS.version),
                             as_text=False)


# Wrap a SavedModel around ProtoBuf
# Necessary for using the tensorflow-serving RESTful API
def build_saved_model():
    builder = tf.saved_model.builder.SavedModelBuilder(FLAGS.serve_dir +
                                                       "/" +
                                                       str(FLAGS.version))

    # Read in ProtoBuf file
    with tf.gfile.GFile(FLAGS.protobuf_dir +
                        "/" +
                        FLAGS.model_name +
                        "_v" +
                        str(FLAGS.version),
                        "rb") as protobuf_file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(protobuf_file.read())

    # Get input and output from graph
    [inp, out] = tf.import_graph_def(graph_def,
                                     name="",
                                     return_elements=["input_bytes:0",
                                                      "output_bytes:0"])

    with tf.Session(graph=out.graph) as sess:
        # Sig def explodes if out_bytes has no shape info
        # Fix: turn it into a batch of 1 image, rather than a single image
        out = tf.expand_dims(out, 0)

        # Build prototypes of input and output
        input_bytes = tf.saved_model.utils.build_tensor_info(inp)
        output_bytes = tf.saved_model.utils.build_tensor_info(out)

        # Create signature for prediction
        signature_definition = tf.saved_model.signature_def_utils.build_signature_def(  # nopep8
            inputs={"input_bytes": input_bytes},
            outputs={"output_bytes": output_bytes},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        # Add meta-information
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.
                DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
            })

    # Create the SavedModel
    builder.save()


def main(_):
    print("Exporting model to ProtoBuf...")
    export_graph()
    print("Wrapping ProtoBuf in SavedModel...")
    build_saved_model()
    print("Exported successfully!")
    print("""Run the server with:
          tensorflow_model_server --rest_api_port=8501 """
          "--model_name=saved_model --model_base_path=$(path)")


if __name__ == "__main__":
    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Establish default arguments
    parser.add_argument("--checkpoint_dir",
                        type=str,
                        default="../CycleGAN-TensorFlow/"
                                "checkpoints/20180628-1208",
                        help="Path to checkpoints directory")

    parser.add_argument("--protobuf_dir",
                        type=str,
                        default="../CycleGAN-TensorFlow/protobufs",
                        help="Path to protobufs directory")

    parser.add_argument("--model_name",
                        type=str,
                        default="model",
                        help="Model name")

    parser.add_argument("--serve_dir",
                        type=str,
                        default="serve",
                        help="Path to serve directory")

    parser.add_argument("--version",
                        type=int,
                        default=1,
                        help="Model version number")

    parser.add_argument("--image_size",
                        type=int,
                        default=64,
                        help="Image size")

    # Parse known arguments
    FLAGS, unparsed = parser.parse_known_args()

    # Run the tensorflow app
    tf.app.run(argv=[sys.argv[0]] + unparsed)
