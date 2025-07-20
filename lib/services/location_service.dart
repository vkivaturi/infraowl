import 'package:geolocator/geolocator.dart';
import 'package:geocoding/geocoding.dart';

class LocationService {
  static LocationService? _instance;
  static LocationService get instance => _instance ??= LocationService._();
  LocationService._();

  Future<Position?> getCurrentLocation() async {
    try {
      // Check if location services are enabled
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        throw Exception('Location services are disabled');
      }

      // Check permissions
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          throw Exception('Location permission denied');
        }
      }

      if (permission == LocationPermission.deniedForever) {
        throw Exception('Location permission permanently denied');
      }

      // Get current position
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );

      return position;
    } catch (e) {
      rethrow;
    }
  }

  String formatPosition(Position position) {
    return '${position.latitude.toStringAsFixed(6)}, ${position.longitude.toStringAsFixed(6)}';
  }

  Future<bool> isLocationServiceEnabled() async {
    return await Geolocator.isLocationServiceEnabled();
  }

  Future<LocationPermission> checkPermission() async {
    return await Geolocator.checkPermission();
  }

  Future<LocationPermission> requestPermission() async {
    return await Geolocator.requestPermission();
  }

  Future<String?> getAddressFromCoordinates(double latitude, double longitude) async {
    try {
      List<Placemark> placemarks = await placemarkFromCoordinates(latitude, longitude);
      if (placemarks.isNotEmpty) {
        Placemark place = placemarks[0];
        return _formatAddress(place);
      }
      return null;
    } catch (e) {
      return null;
    }
  }

  String _formatAddress(Placemark place) {
    List<String> addressComponents = [];
    
    // Street-level details
    if (place.subThoroughfare?.isNotEmpty == true && place.thoroughfare?.isNotEmpty == true) {
      // House/building number + street name
      addressComponents.add('${place.subThoroughfare} ${place.thoroughfare}');
    } else if (place.thoroughfare?.isNotEmpty == true) {
      // Just street name if no house number
      addressComponents.add(place.thoroughfare!);
    } else if (place.street?.isNotEmpty == true) {
      // Fallback to general street info
      addressComponents.add(place.street!);
    }
    
    // Neighborhood/area details
    if (place.subLocality?.isNotEmpty == true) {
      addressComponents.add(place.subLocality!);
    }
    
    // City/town
    if (place.locality?.isNotEmpty == true) {
      addressComponents.add(place.locality!);
    }
    
    // State/province and postal code on same line
    List<String> statePostal = [];
    if (place.administrativeArea?.isNotEmpty == true) {
      statePostal.add(place.administrativeArea!);
    }
    if (place.postalCode?.isNotEmpty == true) {
      statePostal.add(place.postalCode!);
    }
    if (statePostal.isNotEmpty) {
      addressComponents.add(statePostal.join(' '));
    }
    
    return addressComponents.join(', ');
  }

  // Get detailed location information for debugging/development
  Map<String, String?> getDetailedLocationInfo(Placemark place) {
    return {
      'Name': place.name,
      'Street Number': place.subThoroughfare,
      'Street Name': place.thoroughfare,
      'Street (General)': place.street,
      'Neighborhood': place.subLocality,
      'City': place.locality,
      'Sub-Admin Area': place.subAdministrativeArea,
      'State/Province': place.administrativeArea,
      'Postal Code': place.postalCode,
      'Country': place.country,
      'ISO Country Code': place.isoCountryCode,
    };
  }

  Future<double> distanceBetween(
    double startLatitude,
    double startLongitude,
    double endLatitude,
    double endLongitude,
  ) async {
    return Geolocator.distanceBetween(
      startLatitude,
      startLongitude,
      endLatitude,
      endLongitude,
    );
  }
}