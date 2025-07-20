import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EfficientNet Lite Demo',
      home: const ImageClassificationPage(),
    );
  }
}

class ImageClassificationPage extends StatefulWidget {
  const ImageClassificationPage({super.key});
  @override
  State<ImageClassificationPage> createState() => _ImageClassificationPageState();
}

class _ImageClassificationPageState extends State<ImageClassificationPage> {
  late Interpreter _interpreter;
  late List<String> _labels;
  File? _image;
  List<Map<String, dynamic>> _results = [];

  @override
  void initState() {
    super.initState();
    loadModelAndLabels();
  }

  Future<void> loadModelAndLabels() async {
    _interpreter = await Interpreter.fromAsset('assets/2.tflite');
    _labels = await loadLabels('assets/labels.txt');
    setState(() {});
  }

  Future<List<String>> loadLabels(String path) async {
    final rawString = await rootBundle.loadString(path);
    return rawString.trim().split('\n').where((line) => line.isNotEmpty).toList();
  }

  bool _isModelLoaded() {
    try {
      return _labels.isNotEmpty;
    } catch (e) {
      return false;
    }
  }

  Future<void> pickAndClassifyImage() async {
    // Check if model and labels are loaded
    if (!_isModelLoaded()) return;
    
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);
    if (pickedFile == null) return;

    final image = File(pickedFile.path);
    setState(() {
      _image = image;
      _results = [];
    });

    // Read and preprocess
    final rawBytes = await image.readAsBytes();
    img.Image? oriImage = img.decodeImage(rawBytes);
    img.Image resizedImage = img.copyResize(oriImage!, width: 224, height: 224);

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
    _interpreter.run(input, output);

    // Process output
    var scores = output[0];
    List<Map<String, dynamic>> results = [];
    for (int i = 0; i < _labels.length; i++) {
      results.add({'label': _labels[i], 'confidence': scores[i]});
    }
    results.sort((a, b) => (b['confidence'] as num).compareTo(a['confidence'] as num));

    setState(() {
      _results = results.take(5).toList();
    });
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('EfficientNet Lite Demo')),
      body: SingleChildScrollView(
        child: Column(
          children: [
            if (_image != null) Image.file(_image!),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: pickAndClassifyImage,
              child: const Text('Take Photo 1 & Classify'),
            ),
            const SizedBox(height: 16),
            ..._results.map((e) => Text(
                  "${e['label']} - ${(e['confidence']).toStringAsFixed(2)}%",
                  style: const TextStyle(fontSize: 16),
                )),
          ],
        ),
      ),
    );
  }
}