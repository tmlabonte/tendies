# Tendies: TensorFlow Distributed Image Serving
A lightweight, RESTful remote inference library for decoupling deep learning development and application.
Includes CycleGAN and the TensorFlow Object Detection Faster R-CNN as usage examples.

## Motivation
A major challenge in combining deep learning and software engineering as we move towards [Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35) is that deep learning _development_ and _application_ are tightly coupled. This restricts deployment of models in several ways, for example:

* A model developed using Google Cloud Platform (GCP) or Amazon Web Services (AWS) cannot be deployed on a classified network, or trained using classified data.
* A model with a deep, complex architecture cannot be utilized for local inference on low-power Internet of Things (IoT) devices.
* A model with atypical input/output requirements, such as the TensorFlow Object Detection API, cannot be easily integrated with larger, pre-existing software systems.

Tendies solves this problem by acting as a decoupling between these two domains. In the given scenarios:

* Tendies provides an encapsulated TensorFlow-Serving environment that does not require the Cloud for remote inference.
* Tendies decouples the IoT client and the GPU backend, so inference can be run on a powerful machine, then transmitted to the device.
* Tendies enforces input/output standardization through a RESTful API, encapsulating the model server into an easily-integrable black box architecture.

In a sentence, Tendies allows _any_ deep learning model to be embedded in _any_ application.

## Basic usage steps for a custom localhost server:
1. Clone the repository to your machine.
2. Follow the instructions in full_functionality/tendies-extension-tutorial.ipynb to export your model.
3. Build your server with `python ServerBuilder.py --checkpoint_dir $(path) --image_size $(size)`
4. Run the server from bash with `tensorflow_model_server --rest_api_port=8501 --model_name=saved_model --model_base_path=$(path)`
   * You may have to `pip install tensorflow-serving-api-python3`
5. Select or write a client, then do inference on a directory of images with `python client.py --input_dir $(path)`

#### Please see minimum_working_example/tendies-basic-tutorial.ipynb for a short walkthrough of how this library works.
#### Please also view full_functionality/tendies-extension-tutorial.ipynb for instructions on configuring Tendies with your specific use case.
