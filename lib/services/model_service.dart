import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ModelService {
  static ModelService? _instance;
  static ModelService get instance => _instance ??= ModelService._();
  ModelService._();

  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isInitialized = false;

  bool get isModelLoaded => _isInitialized && _labels.isNotEmpty;

  Future<void> loadModelAndLabels() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/2.tflite');
      _labels = await _loadLabels('assets/labels.txt');
      _isInitialized = true;
    } catch (e) {
      _isInitialized = false;
      rethrow;
    }
  }

  Future<List<String>> _loadLabels(String path) async {
    final rawString = await rootBundle.loadString(path);
    return rawString.trim().split('\n').where((line) => line.isNotEmpty).toList();
  }

  Future<List<Map<String, dynamic>>> classifyImage(File imageFile) async {
    if (!isModelLoaded || _interpreter == null) {
      throw Exception('Model not loaded');
    }

    // Read and preprocess
    final rawBytes = await imageFile.readAsBytes();
    img.Image? oriImage = img.decodeImage(rawBytes);
    if (oriImage == null) {
      throw Exception('Could not decode image');
    }
    
    img.Image resizedImage = img.copyResize(oriImage, width: 224, height: 224);

    // Convert image to input tensor
    // EfficientNet Lite0 (int8) usually expects shape: [1, 224, 224, 3] with uint8 data
    var input = List.generate(
        1,
        (_) => List.generate(
            224,
            (y) => List.generate(
                224,
                (x) {
                  final pixel = resizedImage.getPixel(x, y);
                  return [
                    pixel.r,
                    pixel.g,
                    pixel.b
                  ];
                })));

    // Output: shape [1, labels.length]
    var output = List.filled(1 * _labels.length, 0).reshape([1, _labels.length]);

    // Run inference
    _interpreter!.run(input, output);

    // Process output
    var scores = output[0];
    List<Map<String, dynamic>> results = [];
    for (int i = 0; i < _labels.length; i++) {
      results.add({'label': _labels[i], 'confidence': scores[i]});
    }
    results.sort((a, b) => (b['confidence'] as num).compareTo(a['confidence'] as num));

    return results.take(5).toList();
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }
}