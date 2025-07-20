import 'dart:io';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'services/model_service.dart';
import 'services/camera_service.dart';
import 'services/location_service.dart';
import 'services/share_service.dart';

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
  File? _image;
  List<Map<String, dynamic>> _results = [];
  Position? _currentPosition;
  String _locationAddress = 'Getting location...';
  String? _readableAddress;
  final TextEditingController _additionalDetailsController = TextEditingController();
  bool _isLoadingLocation = false;
  bool _isProcessingImage = false;

  // Service instances
  final ModelService _modelService = ModelService.instance;
  final CameraService _cameraService = CameraService.instance;
  final LocationService _locationService = LocationService.instance;
  final ShareService _shareService = ShareService.instance;

  @override
  void initState() {
    super.initState();
    _initializeServices();
    _getCurrentLocation();
  }

  Future<void> _initializeServices() async {
    try {
      await _modelService.loadModelAndLabels();
      setState(() {});
    } catch (e) {
      _showErrorSnackBar('Failed to load AI model: $e');
    }
  }

  Future<void> pickAndClassifyImage() async {
    if (!_modelService.isModelLoaded) {
      _showErrorSnackBar('AI model not loaded yet');
      return;
    }

    try {
      final image = await _cameraService.pickImageFromCamera();
      if (image != null) {
        await _classifyImage(image);
      }
    } catch (e) {
      _showErrorSnackBar('Failed to capture image: $e');
    }
  }

  Future<void> pickFromGalleryAndClassify() async {
    if (!_modelService.isModelLoaded) {
      _showErrorSnackBar('AI model not loaded yet');
      return;
    }

    try {
      final image = await _cameraService.pickImageFromGallery();
      if (image != null) {
        await _classifyImage(image);
      }
    } catch (e) {
      _showErrorSnackBar('Failed to select image: $e');
    }
  }

  Future<void> _classifyImage(File image) async {
    setState(() {
      _image = image;
      _results = [];
      _isProcessingImage = true;
    });

    try {
      final results = await _modelService.classifyImage(image);
      setState(() {
        _results = results;
        _isProcessingImage = false;
      });
    } catch (e) {
      setState(() {
        _isProcessingImage = false;
      });
      _showErrorSnackBar('Failed to classify image: $e');
    }
  }

  Future<void> _getCurrentLocation() async {
    setState(() {
      _isLoadingLocation = true;
    });

    try {
      final position = await _locationService.getCurrentLocation();
      if (position != null) {
        setState(() {
          _currentPosition = position;
          _locationAddress = _locationService.formatPosition(position);
          _readableAddress = null; // Reset address while fetching
        });

        // Get readable address
        final address = await _locationService.getAddressFromCoordinates(
          position.latitude,
          position.longitude,
        );
        
        setState(() {
          _readableAddress = address;
          _isLoadingLocation = false;
        });
      }
    } catch (e) {
      setState(() {
        _locationAddress = 'Error getting location: $e';
        _readableAddress = null;
        _isLoadingLocation = false;
      });
    }
  }

  Future<void> _shareReport() async {
    if (_results.isEmpty) {
      _showErrorSnackBar('Please take a photo first');
      return;
    }

    try {
      final locationInfo = _readableAddress != null 
          ? '$_locationAddress\n$_readableAddress' 
          : _locationAddress;

      await _shareService.shareReport(
        detectedIssues: _results,
        location: locationInfo,
        additionalDetails: _additionalDetailsController.text,
        imageFile: _image,
      );
    } catch (e) {
      _showErrorSnackBar('Failed to share report: $e');
    }
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }

  Color _getBadgeColor(double confidence) {
    if (confidence >= 80) {
      return Colors.red[600]!; // High confidence - red
    } else if (confidence >= 60) {
      return Colors.orange[600]!; // Medium confidence - orange
    } else {
      return Colors.blue[600]!; // Low confidence - blue
    }
  }

  @override
  void dispose() {
    _modelService.dispose();
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
              height: 200,
              margin: const EdgeInsets.all(12),
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
                        const SizedBox(height: 8),
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
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessingImage ? null : pickAndClassifyImage,
                      icon: _isProcessingImage 
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.camera_alt),
                      label: const Text('Camera'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 10),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _isProcessingImage ? null : pickFromGalleryAndClassify,
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Gallery'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.purple[600],
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 10),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Detected Issues section
            if (_results.isNotEmpty) ...[
              Container(
                margin: const EdgeInsets.fromLTRB(12, 8, 12, 8),
                padding: const EdgeInsets.all(12),
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
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      runSpacing: 6,
                      children: _results.take(3).map((result) => Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: _getBadgeColor(result['confidence']),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: _getBadgeColor(result['confidence']).withOpacity(0.3),
                            width: 1,
                          ),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Container(
                              width: 6,
                              height: 6,
                              decoration: const BoxDecoration(
                                color: Colors.white,
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 6),
                            Flexible(
                              child: Text(
                                '${result['label']} ${(result['confidence']).toStringAsFixed(0)}%',
                                style: const TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.white,
                                ),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      )).toList(),
                    ),
                  ],
                ),
              ),
            ],

            // Location section
            Container(
              margin: const EdgeInsets.fromLTRB(12, 0, 12, 8),
              padding: const EdgeInsets.all(12),
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
                  const SizedBox(height: 8),
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
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              _locationAddress,
                              style: const TextStyle(
                                fontSize: 12,
                                fontFamily: 'monospace',
                                color: Colors.grey,
                              ),
                            ),
                            if (_readableAddress != null) ...[
                              const SizedBox(height: 4),
                              Text(
                                _readableAddress!,
                                style: const TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ],
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
              margin: const EdgeInsets.fromLTRB(12, 0, 12, 8),
              padding: const EdgeInsets.all(12),
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
                  const SizedBox(height: 8),
                  TextField(
                    controller: _additionalDetailsController,
                    maxLines: 2,
                    decoration: const InputDecoration(
                      hintText: 'Describe the issue...',
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.all(12),
                    ),
                  ),
                ],
              ),
            ),

            // Share button
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 0, 12, 16),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _shareReport,
                  icon: const Icon(Icons.share),
                  label: const Text('Share Report'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    textStyle: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}