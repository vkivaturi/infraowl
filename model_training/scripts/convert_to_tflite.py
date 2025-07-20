#!/usr/bin/env python3
"""
TensorFlow Lite conversion script for InfraOwl models.

This script converts trained TensorFlow models to TensorFlow Lite format
with optimization for mobile deployment.
"""

import os
import yaml
import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf


class TFLiteConverter:
    def __init__(self, config_path="configs/training_config.yaml"):
        """Initialize the TFLite converter with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = Path(self.config['paths']['models'])
        self.tflite_dir = self.models_dir / "tflite_models"
        self.tflite_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path=None):
        """Load the trained model."""
        if model_path is None:
            model_path = self.models_dir / "saved_models" / "infraowl_model.h5"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"📂 Loading model from {model_path}")
        model = tf.keras.models.load_model(str(model_path))
        return model
    
    def create_representative_dataset(self):
        """Create representative dataset for quantization."""
        processed_data_path = Path(self.config['paths']['processed_data'])
        train_data_path = processed_data_path / "train"
        
        if not train_data_path.exists():
            print("⚠️  Training data not found. Skipping representative dataset.")
            return None
        
        # Load a small sample of training images
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(
            rescale=self.config['data']['preprocessing']['rescale']
        )
        
        target_size = tuple(self.config['data']['preprocessing']['target_size'])
        
        generator = datagen.flow_from_directory(
            train_data_path,
            target_size=target_size,
            batch_size=1,
            class_mode='categorical',
            shuffle=True
        )
        
        def representative_data_gen():
            for i in range(min(100, generator.samples)):  # Use up to 100 samples
                data = next(generator)
                yield [data[0].astype(np.float32)]
        
        return representative_data_gen
    
    def convert_to_tflite(self, model, quantization_type="dynamic"):
        """Convert model to TensorFlow Lite format."""
        print(f"🔄 Converting model to TFLite with {quantization_type} quantization...")
        
        # Initialize converter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimizations
        optimizations = []
        for opt in self.config['tflite']['optimization']:
            if opt == "DEFAULT":
                optimizations.append(tf.lite.Optimize.DEFAULT)
            elif opt == "OPTIMIZE_FOR_SIZE":
                optimizations.append(tf.lite.Optimize.OPTIMIZE_FOR_SIZE)
            elif opt == "OPTIMIZE_FOR_LATENCY":
                optimizations.append(tf.lite.Optimize.OPTIMIZE_FOR_LATENCY)
        
        converter.optimizations = optimizations
        
        # Configure quantization
        if quantization_type == "dynamic":
            # Dynamic range quantization (default)
            pass
        elif quantization_type == "int8":
            # Full integer quantization
            converter.representative_dataset = self.create_representative_dataset()
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif quantization_type == "float16":
            # Float16 quantization
            converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        try:
            tflite_model = converter.convert()
            return tflite_model
        except Exception as e:
            print(f"❌ Conversion failed: {e}")
            raise
    
    def save_tflite_model(self, tflite_model, filename):
        """Save TensorFlow Lite model to file."""
        output_path = self.tflite_dir / filename
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"💾 TFLite model saved to {output_path}")
        
        # Print model size
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"📏 Model size: {size_mb:.2f} MB")
        
        return output_path
    
    def test_tflite_model(self, tflite_path):
        """Test the TensorFlow Lite model."""
        print(f"🧪 Testing TFLite model: {tflite_path}")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("📋 Model Details:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input type: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output type: {output_details[0]['dtype']}")
        
        # Test with random input
        input_shape = input_details[0]['shape']
        input_data = np.random.random_sample(input_shape).astype(input_details[0]['dtype'])
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✅ Test successful! Output shape: {output_data.shape}")
        
        return True
    
    def benchmark_model(self, tflite_path, num_runs=10):
        """Benchmark the TensorFlow Lite model."""
        print(f"⏱️  Benchmarking model ({num_runs} runs)...")
        
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_data = np.random.random_sample(input_shape).astype(input_details[0]['dtype'])
        
        import time
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times) * 1000  # Convert to milliseconds
        std_time = np.std(times) * 1000
        
        print(f"📊 Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        return avg_time, std_time
    
    def convert_all_variants(self, model):
        """Convert model to multiple TFLite variants."""
        variants = []
        
        # Dynamic quantization (default)
        print("\n🔄 Creating dynamic quantized model...")
        dynamic_model = self.convert_to_tflite(model, "dynamic")
        dynamic_path = self.save_tflite_model(dynamic_model, "infraowl_dynamic.tflite")
        variants.append(("dynamic", dynamic_path))
        
        # Float16 quantization
        if self.config['tflite']['quantization']['type'] in ["float16", "all"]:
            try:
                print("\n🔄 Creating float16 quantized model...")
                float16_model = self.convert_to_tflite(model, "float16")
                float16_path = self.save_tflite_model(float16_model, "infraowl_float16.tflite")
                variants.append(("float16", float16_path))
            except Exception as e:
                print(f"⚠️  Float16 conversion failed: {e}")
        
        # INT8 quantization (if representative dataset available)
        if self.config['tflite']['quantization']['type'] in ["int8", "all"]:
            try:
                print("\n🔄 Creating INT8 quantized model...")
                int8_model = self.convert_to_tflite(model, "int8")
                int8_path = self.save_tflite_model(int8_model, "infraowl_int8.tflite")
                variants.append(("int8", int8_path))
            except Exception as e:
                print(f"⚠️  INT8 conversion failed: {e}")
        
        return variants
    
    def run_conversion(self, model_path=None):
        """Run the complete conversion pipeline."""
        print("🚀 Starting TensorFlow Lite conversion...")
        
        try:
            # Load model
            model = self.load_model(model_path)
            
            # Convert to different variants
            variants = self.convert_all_variants(model)
            
            # Test and benchmark each variant
            print("\n🧪 Testing converted models...")
            for variant_name, variant_path in variants:
                print(f"\n--- {variant_name.upper()} Variant ---")
                self.test_tflite_model(variant_path)
                self.benchmark_model(variant_path)
            
            # Copy best model to assets (for the current model, use dynamic)
            best_model_path = next(path for name, path in variants if name == "dynamic")
            assets_path = Path("../assets/2.tflite")
            assets_path.parent.mkdir(exist_ok=True)
            
            import shutil
            shutil.copy2(best_model_path, assets_path)
            print(f"\n📋 Best model copied to Flutter assets: {assets_path}")
            
            print("\n✅ TensorFlow Lite conversion completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during conversion: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Convert InfraOwl model to TensorFlow Lite')
    parser.add_argument('--config', 
                       default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--model', 
                       help='Path to trained model (default: from config)')
    
    args = parser.parse_args()
    
    converter = TFLiteConverter(args.config)
    converter.run_conversion(args.model)


if __name__ == "__main__":
    main()