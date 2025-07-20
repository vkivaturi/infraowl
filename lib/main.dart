import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:geolocator/geolocator.dart';
import 'package:share_plus/share_plus.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'InfraOwl - Report Issues',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
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
  Position? _currentPosition;
  String _locationAddress = 'Getting location...';
  final TextEditingController _additionalDetailsController = TextEditingController();
  bool _isLoadingLocation = false;

  @override
  void initState() {
    super.initState();
    loadModelAndLabels();
    _getCurrentLocation();
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

    await _classifyImage(File(pickedFile.path));
  }

  Future<void> pickFromGalleryAndClassify() async {
    // Check if model and labels are loaded
    if (!_isModelLoaded()) return;
    
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile == null) return;

    await _classifyImage(File(pickedFile.path));
  }

  Future<void> _classifyImage(File image) async {
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

  Future<void> _getCurrentLocation() async {
    setState(() {
      _isLoadingLocation = true;
    });

    try {
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        setState(() {
          _locationAddress = 'Location services are disabled';
          _isLoadingLocation = false;
        });
        return;
      }

      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          setState(() {
            _locationAddress = 'Location permission denied';
            _isLoadingLocation = false;
          });
          return;
        }
      }

      if (permission == LocationPermission.deniedForever) {
        setState(() {
          _locationAddress = 'Location permission permanently denied';
          _isLoadingLocation = false;
        });
        return;
      }

      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      setState(() {
        _currentPosition = position;
        _locationAddress = '${position.latitude.toStringAsFixed(6)}, ${position.longitude.toStringAsFixed(6)}';
        _isLoadingLocation = false;
      });
    } catch (e) {
      setState(() {
        _locationAddress = 'Error getting location: $e';
        _isLoadingLocation = false;
      });
    }
  }

  Future<void> _shareReport() async {
    if (_results.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please take a photo first')),
      );
      return;
    }

    String detectedIssues = _results.map((r) => 
      '${r['label']}: ${(r['confidence']).toStringAsFixed(1)}%'
    ).join('\n');

    String shareText = '''
InfraOwl Issue Report

Detected Issues:
$detectedIssues

Location: $_locationAddress

Additional Details:
${_additionalDetailsController.text.isEmpty ? 'None' : _additionalDetailsController.text}

Generated with InfraOwl App
''';

    if (_image != null) {
      await Share.shareXFiles([XFile(_image!.path)], text: shareText);
    } else {
      await Share.share(shareText);
    }
  }

  @override
  void dispose() {
    _interpreter.close();
    _additionalDetailsController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: const Text('Report Issue'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 1,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {}, // Placeholder for settings
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Image section
            Container(
              width: double.infinity,
              height: 250,
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.2),
                    spreadRadius: 2,
                    blurRadius: 5,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: _image != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.file(
                        _image!,
                        fit: BoxFit.cover,
                        width: double.infinity,
                        height: double.infinity,
                      ),
                    )
                  : Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.camera_alt,
                          size: 80,
                          color: Colors.grey[400],
                        ),
                        const SizedBox(height: 16),
                        const Text(
                          'Take a photo to report an issue',
                          style: TextStyle(
                            fontSize: 16,
                            color: Colors.grey,
                          ),
                        ),
                      ],
                    ),
            ),

            // Camera buttons
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: pickAndClassifyImage,
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Camera'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: pickFromGalleryAndClassify,
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Gallery'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.grey[600],
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Detected Issues section
            if (_results.isNotEmpty) ...[
              Container(
                margin: const EdgeInsets.all(16),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.2),
                      spreadRadius: 2,
                      blurRadius: 5,
                      offset: const Offset(0, 3),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Text(
                          'Detected Issues',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const Spacer(),
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 8,
                            vertical: 4,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.orange[100],
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: const Text(
                            'AI Processing',
                            style: TextStyle(
                              color: Colors.orange,
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    ..._results.take(3).map((result) => Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Row(
                            children: [
                              Container(
                                width: 8,
                                height: 8,
                                decoration: const BoxDecoration(
                                  color: Colors.red,
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 12),
                              Expanded(
                                child: Text(
                                  result['label'],
                                  style: const TextStyle(fontSize: 16),
                                ),
                              ),
                              Text(
                                '${(result['confidence']).toStringAsFixed(1)}%',
                                style: TextStyle(
                                  fontSize: 14,
                                  color: Colors.grey[600],
                                ),
                              ),
                            ],
                          ),
                        )),
                  ],
                ),
              ),
            ],

            // Location section
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.2),
                    spreadRadius: 2,
                    blurRadius: 5,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Row(
                    children: [
                      Icon(Icons.location_on, color: Colors.red),
                      SizedBox(width: 8),
                      Text(
                        'Location',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      if (_isLoadingLocation)
                        const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      else
                        const Icon(Icons.my_location, size: 16),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          _locationAddress,
                          style: const TextStyle(fontSize: 14),
                        ),
                      ),
                      TextButton(
                        onPressed: _getCurrentLocation,
                        child: const Text('Refresh'),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // Additional Details section
            Container(
              margin: const EdgeInsets.all(16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.2),
                    spreadRadius: 2,
                    blurRadius: 5,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Additional Details',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  TextField(
                    controller: _additionalDetailsController,
                    maxLines: 3,
                    decoration: const InputDecoration(
                      hintText: 'Describe the issue...',
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.all(12),
                    ),
                  ),
                ],
              ),
            ),

            // Share Report section
            Container(
              margin: const EdgeInsets.all(16),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.2),
                    spreadRadius: 2,
                    blurRadius: 5,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Share Report',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _shareReport,
                          icon: const Icon(Icons.share),
                          label: const Text('Social'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.blue,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _shareReport,
                          icon: const Icon(Icons.message),
                          label: const Text('Message'),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.green,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 12),
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}