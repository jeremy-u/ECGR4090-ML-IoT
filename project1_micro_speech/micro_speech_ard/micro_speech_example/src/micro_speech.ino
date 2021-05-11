/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 64 * 1024; // 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace
int g_loop_counter = 0;
// The name of this function is important for Arduino compatibility.
void setup() {
  
  delay(5000);
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  TF_LITE_REPORT_ERROR(error_reporter, "Starting Arduino");  
  TF_LITE_REPORT_ERROR(error_reporter, "kFeatureElementCount = %d", kFeatureElementCount);  
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d, supported version: %d",
                         model->version(), TFLITE_SCHEMA_VERSION);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: About to add DWConv2D");
  static tflite::MicroMutableOpResolver<8> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: About to AddFullyConnected");
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: About to AddSoftmax");
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: About to  AddReshape");
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: About to AddMaxPool2D");
  if (micro_op_resolver.AddMaxPool2D() != kTfLiteOk) {
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: About to AddConv2D");
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
    TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: About to AddAdd");
  if (micro_op_resolver.AddAdd() != kTfLiteOk) {
    return;
  }
    TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: About to AddMul");
  if (micro_op_resolver.AddMul() != kTfLiteOk) {
    return;
  }
  
  
  TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: Done adding resolvers");
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: Interpreter constructed");
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  TF_LITE_REPORT_ERROR(error_reporter, "JHDBG: Allocated Tensors");

  
  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || 
      (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceCount) || 
      (model_input->dims->data[2] !=kFeatureSliceSize) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Still Bad input tensor parameters in model");
  ///////
    TF_LITE_REPORT_ERROR(error_reporter, "input->dims->size: %d", model_input->dims->size);
    TF_LITE_REPORT_ERROR(error_reporter, "input->dims[0]: %d", model_input->dims->data[0]);
    TF_LITE_REPORT_ERROR(error_reporter, "input->dims[1]: %d", model_input->dims->data[1]);
    TF_LITE_REPORT_ERROR(error_reporter, "input->dims[2]: %d", model_input->dims->data[2]);
    TF_LITE_REPORT_ERROR(error_reporter, "input->dims[3]: %d", model_input->dims->data[3]);
    TF_LITE_REPORT_ERROR(error_reporter, "Feature Slice: %d x %d", kFeatureSliceCount,  kFeatureSliceSize);
    TF_LITE_REPORT_ERROR(error_reporter, "Input Type is: %d", model_input->type);

  if (model_input->dims->size != 4)
    TF_LITE_REPORT_ERROR(error_reporter, "dims size != 4");
  if (model_input->dims->data[0] != 1)
    TF_LITE_REPORT_ERROR(error_reporter, "dims[0] != 1");
  if (model_input->dims->data[1] != kFeatureSliceCount)
    TF_LITE_REPORT_ERROR(error_reporter, "dims[1] != SliceCount");
  if (model_input->dims->data[2] !=kFeatureSliceSize)
    TF_LITE_REPORT_ERROR(error_reporter, "dims[2] != kFeatureSliceSize");
  if (model_input->type != kTfLiteInt8)
    TF_LITE_REPORT_ERROR(error_reporter, "type not Int8");
  ////////
    return;
  }
  
  model_input_buffer = model_input->data.int8;

  TF_LITE_REPORT_ERROR(error_reporter,
                         "About to create feature provider, count=%d", kFeatureElementCount);
  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  TF_LITE_REPORT_ERROR(error_reporter, "created feature_provider, size=%d", static_feature_provider.feature_size_);
  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
  TF_LITE_REPORT_ERROR(error_reporter, "Done with setup()");

}

// The name of this function is important for Arduino compatibility.
void loop() {
  g_loop_counter += 1;

  // TF_LITE_REPORT_ERROR(error_reporter, "Starting loop %d", g_loop_counter);
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    delay(1000);  // delay 1s so these errors don't pile up too fast
    return;
  }
  // TF_LITE_REPORT_ERROR(error_reporter, "Got %d new slices of features", how_many_new_slices);
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }
  
  // Copy feature buffer to input tensor
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  int time_before_invoke, time_after_invoke;
  time_before_invoke=micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  time_after_invoke=micros();
  /* TF_LITE_REPORT_ERROR(error_reporter, "%d => %d ; Invoke took %d us", 
                         time_before_invoke, time_after_invoke, 
                         time_after_invoke-time_before_invoke);
  */
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }
  // else {
  //   TF_LITE_REPORT_ERROR(error_reporter, "Invoke succeeded");
  // }
  
  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);

  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // TF_LITE_REPORT_ERROR(error_reporter, "About to Respond To Command");
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  RespondToCommand(error_reporter, current_time, found_command, score,
                   is_new_command);

  
/******* To print out activations scores
  int8_t* output_dat = output->data.int8;
  char msg_str[50];
  sprintf(msg_str, "%d,%d,%d,%d", output_dat[0], output_dat[1], output_dat[2], output_dat[3]);
  Serial.println(msg_str);
  *******/  
  // TF_LITE_REPORT_ERROR(error_reporter, "Just did ProcessLatestResults");
}
