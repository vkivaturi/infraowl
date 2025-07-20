import 'dart:io';
import 'package:share_plus/share_plus.dart';

class ShareService {
  static ShareService? _instance;
  static ShareService get instance => _instance ??= ShareService._();
  ShareService._();

  Future<void> shareReport({
    required List<Map<String, dynamic>> detectedIssues,
    required String location,
    String? additionalDetails,
    File? imageFile,
  }) async {
    try {
      final detectedIssuesText = detectedIssues
          .map((r) => '${r['label']}: ${(r['confidence']).toStringAsFixed(1)}%')
          .join('\n');

      final shareText = '''
InfraOwl Issue Report

Detected Issues:
$detectedIssuesText

Location: $location

Additional Details:
${additionalDetails?.isEmpty == true ? 'None' : additionalDetails ?? 'None'}

Generated with InfraOwl App
''';

      if (imageFile != null) {
        await Share.shareXFiles([XFile(imageFile.path)], text: shareText);
      } else {
        await Share.share(shareText);
      }
    } catch (e) {
      throw Exception('Failed to share report: $e');
    }
  }

  Future<void> shareText(String text) async {
    try {
      await Share.share(text);
    } catch (e) {
      throw Exception('Failed to share text: $e');
    }
  }

  Future<void> shareFile(File file, {String? text}) async {
    try {
      await Share.shareXFiles([XFile(file.path)], text: text);
    } catch (e) {
      throw Exception('Failed to share file: $e');
    }
  }

  Future<void> shareFiles(List<File> files, {String? text}) async {
    try {
      final xFiles = files.map((file) => XFile(file.path)).toList();
      await Share.shareXFiles(xFiles, text: text);
    } catch (e) {
      throw Exception('Failed to share files: $e');
    }
  }
}