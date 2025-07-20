import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

class ImageProcessingService {
  static ImageProcessingService? _instance;
  static ImageProcessingService get instance => _instance ??= ImageProcessingService._();
  ImageProcessingService._();

  Future<img.Image?> loadImage(File imageFile) async {
    try {
      final bytes = await imageFile.readAsBytes();
      return img.decodeImage(bytes);
    } catch (e) {
      throw Exception('Failed to load image: $e');
    }
  }

  img.Image resizeImage(img.Image image, {int width = 224, int height = 224}) {
    return img.copyResize(image, width: width, height: height);
  }

  Future<img.Image?> loadAndResizeImage(File imageFile, {int width = 224, int height = 224}) async {
    final image = await loadImage(imageFile);
    if (image == null) return null;
    return resizeImage(image, width: width, height: height);
  }

  List<List<List<List<int>>>> imageToTensorInput(img.Image image) {
    // Convert image to input tensor format for TensorFlow Lite
    // Shape: [1, height, width, 3] for RGB
    return List.generate(
        1,
        (_) => List.generate(
            image.height,
            (y) => List.generate(
                image.width,
                (x) {
                  final pixel = image.getPixel(x, y);
                  return [
                    pixel.r,
                    pixel.g,
                    pixel.b
                  ];
                })));
  }

  Future<Uint8List> compressImage(File imageFile, {int quality = 85}) async {
    try {
      final image = await loadImage(imageFile);
      if (image == null) throw Exception('Could not decode image');
      
      return Uint8List.fromList(img.encodeJpg(image, quality: quality));
    } catch (e) {
      throw Exception('Failed to compress image: $e');
    }
  }

  Future<File> saveProcessedImage(img.Image image, String outputPath) async {
    try {
      final bytes = img.encodeJpg(image);
      final file = File(outputPath);
      await file.writeAsBytes(bytes);
      return file;
    } catch (e) {
      throw Exception('Failed to save processed image: $e');
    }
  }
}