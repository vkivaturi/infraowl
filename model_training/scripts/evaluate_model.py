#!/usr/bin/env python3
"""
Model evaluation script for InfraOwl infrastructure detection.

This script provides comprehensive evaluation of trained models including:
- Test set evaluation
- Confusion matrix generation
- Classification reports
- Model comparison
"""

import os
import yaml
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ModelEvaluator:
    def __init__(self, config_path="configs/training_config.yaml"):
        """Initialize the model evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = Path(self.config['paths']['models'])
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)
        
        self.class_names = self.config['classes']
        self.setup_test_data()
    
    def setup_test_data(self):
        """Setup test data generator."""
        processed_data_path = Path(self.config['paths']['processed_data'])
        test_data_path = processed_data_path / "test"
        
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_data_path}")
        
        test_datagen = ImageDataGenerator(
            rescale=self.config['data']['preprocessing']['rescale']
        )
        
        target_size = tuple(self.config['data']['preprocessing']['target_size'])
        
        self.test_generator = test_datagen.flow_from_directory(
            test_data_path,
            target_size=target_size,
            batch_size=1,  # Use batch size 1 for easier evaluation
            class_mode='categorical',
            shuffle=False  # Important: don't shuffle for consistent evaluation
        )
        
        print(f"📊 Test samples: {self.test_generator.samples}")
        print(f"📊 Found classes: {list(self.test_generator.class_indices.keys())}")
    
    def load_model(self, model_path=None):
        """Load the trained model."""
        if model_path is None:
            model_path = self.models_dir / "saved_models" / "infraowl_model.h5"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"📂 Loading model from {model_path}")
        model = tf.keras.models.load_model(str(model_path))
        return model
    
    def load_tflite_model(self, tflite_path=None):
        """Load TensorFlow Lite model."""
        if tflite_path is None:
            tflite_path = self.models_dir / "tflite_models" / "infraowl_dynamic.tflite"
        
        if not Path(tflite_path).exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
        
        print(f"📂 Loading TFLite model from {tflite_path}")
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        return interpreter
    
    def predict_keras_model(self, model):
        """Get predictions from Keras model."""
        print("🔮 Getting predictions from Keras model...")
        
        predictions = model.predict(self.test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = self.test_generator.classes
        
        return predictions, predicted_classes, true_classes
    
    def predict_tflite_model(self, interpreter):
        """Get predictions from TensorFlow Lite model."""
        print("🔮 Getting predictions from TFLite model...")
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        true_classes = []
        
        # Reset generator
        self.test_generator.reset()
        
        for i in range(self.test_generator.samples):
            # Get next batch
            batch_data, batch_labels = next(self.test_generator)
            
            # Prepare input
            input_data = batch_data.astype(input_details[0]['dtype'])
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data[0])
            true_classes.append(np.argmax(batch_labels[0]))
        
        predictions = np.array(predictions)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.array(true_classes)
        
        return predictions, predicted_classes, true_classes
    
    def calculate_metrics(self, true_classes, predicted_classes, predictions):
        """Calculate evaluation metrics."""
        # Basic metrics
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_classes, predicted_classes, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_classes, predicted_classes, average=None)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support_per_class
        }
        
        return metrics
    
    def plot_confusion_matrix(self, true_classes, predicted_classes, model_name="Model"):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        filename = f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        plt.savefig(self.outputs_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to {self.outputs_dir}/{filename}")
        return cm
    
    def plot_classification_report(self, true_classes, predicted_classes, model_name="Model"):
        """Generate and save classification report."""
        report = classification_report(
            true_classes, predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better visualization
        import pandas as pd
        df = pd.DataFrame(report).transpose()
        
        # Plot heatmap of metrics
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.iloc[:-3, :-1], annot=True, fmt='.3f', cmap='RdYlBu_r')
        plt.title(f'{model_name} - Classification Report')
        plt.tight_layout()
        
        filename = f'{model_name.lower().replace(" ", "_")}_classification_report.png'
        plt.savefig(self.outputs_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save text report
        text_report = classification_report(
            true_classes, predicted_classes,
            target_names=self.class_names
        )
        
        text_filename = f'{model_name.lower().replace(" ", "_")}_classification_report.txt'
        with open(self.outputs_dir / text_filename, 'w') as f:
            f.write(f"{model_name} Classification Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(text_report)
        
        print(f"📊 Classification report saved to {self.outputs_dir}/{text_filename}")
        return report
    
    def plot_prediction_confidence(self, predictions, true_classes, predicted_classes, model_name="Model"):
        """Plot prediction confidence distribution."""
        # Get confidence scores (max probability for each prediction)
        confidence_scores = np.max(predictions, axis=1)
        
        # Separate correct and incorrect predictions
        correct_mask = (true_classes == predicted_classes)
        correct_confidence = confidence_scores[correct_mask]
        incorrect_confidence = confidence_scores[~correct_mask]
        
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Confidence distribution
        plt.subplot(1, 2, 1)
        plt.hist(correct_confidence, bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_confidence, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title(f'{model_name} - Prediction Confidence')
        plt.legend()
        
        # Plot 2: Confidence vs Accuracy
        plt.subplot(1, 2, 2)
        confidence_bins = np.linspace(0, 1, 11)
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(correct_mask[bin_mask])
                accuracies.append(bin_accuracy)
            else:
                accuracies.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.plot(bin_centers, accuracies, 'o-')
        plt.xlabel('Confidence Score')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name} - Calibration')
        plt.grid(True)
        
        plt.tight_layout()
        filename = f'{model_name.lower().replace(" ", "_")}_confidence_analysis.png'
        plt.savefig(self.outputs_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confidence analysis saved to {self.outputs_dir}/{filename}")
    
    def evaluate_keras_model(self, model_path=None):
        """Evaluate Keras model."""
        print("\n🔍 Evaluating Keras Model...")
        
        model = self.load_model(model_path)
        predictions, predicted_classes, true_classes = self.predict_keras_model(model)
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_classes, predicted_classes, predictions)
        
        # Generate visualizations
        self.plot_confusion_matrix(true_classes, predicted_classes, "Keras Model")
        self.plot_classification_report(true_classes, predicted_classes, "Keras Model")
        self.plot_prediction_confidence(predictions, true_classes, predicted_classes, "Keras Model")
        
        return metrics
    
    def evaluate_tflite_model(self, tflite_path=None):
        """Evaluate TensorFlow Lite model."""
        print("\n🔍 Evaluating TFLite Model...")
        
        interpreter = self.load_tflite_model(tflite_path)
        predictions, predicted_classes, true_classes = self.predict_tflite_model(interpreter)
        
        # Calculate metrics
        metrics = self.calculate_metrics(true_classes, predicted_classes, predictions)
        
        # Generate visualizations
        self.plot_confusion_matrix(true_classes, predicted_classes, "TFLite Model")
        self.plot_classification_report(true_classes, predicted_classes, "TFLite Model")
        self.plot_prediction_confidence(predictions, true_classes, predicted_classes, "TFLite Model")
        
        return metrics
    
    def save_evaluation_summary(self, keras_metrics=None, tflite_metrics=None):
        """Save evaluation summary to file."""
        with open(self.outputs_dir / 'evaluation_summary.txt', 'w') as f:
            f.write("InfraOwl Model Evaluation Summary\n")
            f.write("=" * 40 + "\n\n")
            
            if keras_metrics:
                f.write("Keras Model Performance:\n")
                f.write(f"  Accuracy: {keras_metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {keras_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {keras_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {keras_metrics['f1_score']:.4f}\n\n")
                
                f.write("Per-Class Performance (Keras):\n")
                for i, class_name in enumerate(self.class_names):
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {keras_metrics['precision_per_class'][i]:.4f}\n")
                    f.write(f"    Recall: {keras_metrics['recall_per_class'][i]:.4f}\n")
                    f.write(f"    F1-Score: {keras_metrics['f1_per_class'][i]:.4f}\n")
                    f.write(f"    Support: {keras_metrics['support_per_class'][i]}\n\n")
            
            if tflite_metrics:
                f.write("TFLite Model Performance:\n")
                f.write(f"  Accuracy: {tflite_metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {tflite_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {tflite_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {tflite_metrics['f1_score']:.4f}\n\n")
        
        print(f"📄 Evaluation summary saved to {self.outputs_dir}/evaluation_summary.txt")
    
    def run_evaluation(self, keras_model_path=None, tflite_model_path=None, compare_models=True):
        """Run complete evaluation pipeline."""
        print("🚀 Starting InfraOwl model evaluation...")
        
        keras_metrics = None
        tflite_metrics = None
        
        try:
            # Evaluate Keras model
            if keras_model_path or (self.models_dir / "saved_models" / "infraowl_model.h5").exists():
                keras_metrics = self.evaluate_keras_model(keras_model_path)
                print(f"\n✅ Keras Model Accuracy: {keras_metrics['accuracy']:.4f}")
            
            # Evaluate TFLite model
            if tflite_model_path or (self.models_dir / "tflite_models" / "infraowl_dynamic.tflite").exists():
                tflite_metrics = self.evaluate_tflite_model(tflite_model_path)
                print(f"\n✅ TFLite Model Accuracy: {tflite_metrics['accuracy']:.4f}")
            
            # Compare models if both available
            if compare_models and keras_metrics and tflite_metrics:
                print("\n📊 Model Comparison:")
                print(f"  Keras vs TFLite Accuracy: {keras_metrics['accuracy']:.4f} vs {tflite_metrics['accuracy']:.4f}")
                accuracy_diff = abs(keras_metrics['accuracy'] - tflite_metrics['accuracy'])
                print(f"  Accuracy Difference: {accuracy_diff:.4f}")
                
                if accuracy_diff < 0.01:
                    print("  ✅ Models perform similarly (difference < 1%)")
                else:
                    print("  ⚠️  Significant performance difference detected")
            
            # Save summary
            self.save_evaluation_summary(keras_metrics, tflite_metrics)
            
            print("\n✅ Model evaluation completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Evaluate InfraOwl infrastructure detection model')
    parser.add_argument('--config', 
                       default='configs/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--keras-model', 
                       help='Path to Keras model (default: from config)')
    parser.add_argument('--tflite-model', 
                       help='Path to TFLite model (default: from config)')
    parser.add_argument('--no-comparison', 
                       action='store_true',
                       help='Skip model comparison')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.config)
    evaluator.run_evaluation(
        keras_model_path=args.keras_model,
        tflite_model_path=args.tflite_model,
        compare_models=not args.no_comparison
    )


if __name__ == "__main__":
    main()