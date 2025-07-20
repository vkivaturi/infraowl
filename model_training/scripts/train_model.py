#!/usr/bin/env python3
"""
Model training script for InfraOwl infrastructure detection.

This script handles the complete training pipeline including:
- Model architecture setup
- Data loading and augmentation
- Training with callbacks
- Model evaluation and saving
"""

import os
import yaml
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard
)


class InfraOwlTrainer:
    def __init__(self, config_path="configs/training_config.yaml"):
        """Initialize the trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_directories()
        self.setup_data_generators()
        
    def setup_directories(self):
        """Create necessary directories for training outputs."""
        self.models_dir = Path(self.config['paths']['models'])
        self.logs_dir = Path(self.config['paths']['logs'])
        self.outputs_dir = Path("outputs")
        
        for dir_path in [self.models_dir, self.logs_dir, self.outputs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "saved_models").mkdir(exist_ok=True)
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "tflite_models").mkdir(exist_ok=True)
    
    def setup_data_generators(self):
        """Setup data generators with augmentation."""
        # Training data generator with augmentation
        if self.config['data']['augmentation']['enabled']:
            aug_config = self.config['data']['augmentation']
            train_datagen = ImageDataGenerator(
                rescale=self.config['data']['preprocessing']['rescale'],
                rotation_range=aug_config['rotation_range'],
                width_shift_range=aug_config['width_shift_range'],
                height_shift_range=aug_config['height_shift_range'],
                shear_range=aug_config['shear_range'],
                zoom_range=aug_config['zoom_range'],
                horizontal_flip=aug_config['horizontal_flip'],
                brightness_range=aug_config['brightness_range'],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=self.config['data']['preprocessing']['rescale']
            )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=self.config['data']['preprocessing']['rescale']
        )
        
        # Load data
        processed_data_path = Path(self.config['paths']['processed_data'])
        target_size = tuple(self.config['data']['preprocessing']['target_size'])
        batch_size = self.config['training']['batch_size']
        
        self.train_generator = train_datagen.flow_from_directory(
            processed_data_path / "train",
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            processed_data_path / "validation",
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_generator = val_datagen.flow_from_directory(
            processed_data_path / "test",
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print(f"📊 Training samples: {self.train_generator.samples}")
        print(f"📊 Validation samples: {self.val_generator.samples}")
        print(f"📊 Test samples: {self.test_generator.samples}")
        print(f"📊 Classes found: {list(self.train_generator.class_indices.keys())}")
    
    def create_model(self):
        """Create the model architecture."""
        architecture = self.config['model']['architecture']
        input_shape = tuple(self.config['model']['input_size'])
        num_classes = self.config['model']['num_classes']
        
        print(f"🏗️  Creating {architecture} model...")
        
        if architecture == "efficientnet_lite0":
            # Use EfficientNetB0 as approximation for EfficientNet-Lite0
            base_model = EfficientNetB0(
                weights=self.config['model']['weights'],
                include_top=False,
                input_shape=input_shape
            )
        elif architecture == "mobilenetv3_small":
            base_model = MobileNetV3Small(
                weights=self.config['model']['weights'],
                include_top=False,
                input_shape=input_shape
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Freeze base model layers initially (for transfer learning)
        base_model.trainable = False
        
        # Add custom classification head
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax', name='predictions')
        ])
        
        return model
    
    def compile_model(self, model):
        """Compile the model with optimizer and loss function."""
        optimizer_name = self.config['training']['optimizer']
        learning_rate = self.config['training']['learning_rate']
        
        if optimizer_name == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        model.compile(
            optimizer=optimizer,
            loss=self.config['training']['loss_function'],
            metrics=self.config['training']['metrics']
        )
        
        return model
    
    def setup_callbacks(self):
        """Setup training callbacks."""
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.models_dir / "checkpoints" / "best_model.h5"
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        )
        
        # Early stopping
        if self.config['training']['early_stopping']['monitor']:
            callbacks.append(
                EarlyStopping(
                    monitor=self.config['training']['early_stopping']['monitor'],
                    patience=self.config['training']['early_stopping']['patience'],
                    restore_best_weights=self.config['training']['early_stopping']['restore_best_weights'],
                    verbose=1
                )
            )
        
        # Learning rate reduction
        if self.config['training']['lr_schedule']['type'] == "reduce_on_plateau":
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config['training']['lr_schedule']['factor'],
                    patience=self.config['training']['lr_schedule']['patience'],
                    min_lr=self.config['training']['lr_schedule']['min_lr'],
                    verbose=1
                )
            )
        
        # TensorBoard
        if self.config['logging']['tensorboard']:
            log_dir = self.logs_dir / "tensorboard"
            callbacks.append(
                TensorBoard(
                    log_dir=str(log_dir),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True
                )
            )
        
        return callbacks
    
    def train_model(self, model, callbacks):
        """Train the model."""
        print("🚀 Starting model training...")
        
        history = model.fit(
            self.train_generator,
            epochs=self.config['training']['epochs'],
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model):
        """Evaluate the model on test data."""
        print("📊 Evaluating model on test data...")
        
        test_loss, test_accuracy = model.evaluate(self.test_generator, verbose=1)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self, history):
        """Plot and save training history."""
        if not self.config['logging']['plot_training_history']:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.outputs_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training history plot saved to {self.outputs_dir}/training_history.png")
    
    def save_model(self, model):
        """Save the trained model."""
        model_path = self.models_dir / "saved_models" / "infraowl_model.h5"
        model.save(str(model_path))
        print(f"💾 Model saved to {model_path}")
        
        # Save model summary
        if self.config['logging']['save_model_summary']:
            with open(self.outputs_dir / 'model_summary.txt', 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            print(f"📄 Model summary saved to {self.outputs_dir}/model_summary.txt")
    
    def run_training(self):
        """Run the complete training pipeline."""
        print("🎯 Starting InfraOwl model training pipeline...")
        
        try:
            # Create and compile model
            model = self.create_model()
            model = self.compile_model(model)
            
            print("📋 Model Summary:")
            model.summary()
            
            # Setup callbacks
            callbacks = self.setup_callbacks()
            
            # Train model
            history = self.train_model(model, callbacks)
            
            # Load best model from checkpoint
            best_model_path = self.models_dir / "checkpoints" / "best_model.h5"
            if best_model_path.exists():
                model = keras.models.load_model(str(best_model_path))
                print("✅ Loaded best model from checkpoint")
            
            # Evaluate model
            test_loss, test_accuracy = self.evaluate_model(model)
            
            # Plot training history
            self.plot_training_history(history)
            
            # Save final model
            self.save_model(model)
            
            print("✅ Training completed successfully!")
            print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")
            
            return model, history
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Train InfraOwl infrastructure detection model')
    parser.add_argument('--config', 
                       default='configs/training_config.yaml',
                       help='Path to training configuration file')
    
    args = parser.parse_args()
    
    # Set GPU memory growth to avoid allocation issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    trainer = InfraOwlTrainer(args.config)
    model, history = trainer.run_training()


if __name__ == "__main__":
    main()