#!/usr/bin/env python3
"""
Data preprocessing script for InfraOwl model training.

This script organizes raw images into train/validation/test splits,
resizes images, and prepares the dataset for training.
"""

import os
import shutil
import yaml
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse


class DataPreprocessor:
    def __init__(self, config_path="configs/training_config.yaml"):
        """Initialize the data preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = Path(self.config['paths']['raw_data'])
        self.processed_data_path = Path(self.config['paths']['processed_data'])
        self.target_size = tuple(self.config['data']['preprocessing']['target_size'])
        
        # Create processed data directories
        self.train_dir = self.processed_data_path / "train"
        self.val_dir = self.processed_data_path / "validation"
        self.test_dir = self.processed_data_path / "test"
        
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_raw_data(self):
        """Validate the raw data structure and content."""
        print("🔍 Validating raw data...")
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_data_path}")
        
        classes = self.config['classes']
        stats = {}
        
        for class_name in classes:
            class_dir = self.raw_data_path / class_name
            if not class_dir.exists():
                print(f"⚠️  Warning: Class directory '{class_name}' not found")
                stats[class_name] = 0
                continue
            
            # Count valid images
            valid_images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                valid_images.extend(list(class_dir.glob(ext)))
            
            stats[class_name] = len(valid_images)
            print(f"📁 {class_name}: {len(valid_images)} images")
        
        total_images = sum(stats.values())
        print(f"\n📊 Total images: {total_images}")
        
        if total_images == 0:
            raise ValueError("No valid images found in raw data directories")
        
        # Check for class imbalance
        min_images = min(stats.values())
        max_images = max(stats.values())
        if max_images > 0 and min_images / max_images < 0.5:
            print("⚠️  Warning: Significant class imbalance detected")
            print("   Consider balancing your dataset for better performance")
        
        return stats
    
    def resize_and_save_image(self, image_path, output_path):
        """Resize and save a single image."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize maintaining aspect ratio, then center crop
                img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
                
                # Create new image with target size and paste resized image
                new_img = Image.new('RGB', self.target_size, (0, 0, 0))
                paste_x = (self.target_size[0] - img.width) // 2
                paste_y = (self.target_size[1] - img.height) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                # Save processed image
                new_img.save(output_path, 'JPEG', quality=95)
                return True
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return False
    
    def split_and_process_data(self):
        """Split data into train/val/test and process images."""
        print("\n🔄 Processing and splitting data...")
        
        # Clear existing processed data
        for dir_path in [self.train_dir, self.val_dir, self.test_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        classes = self.config['classes']
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        processed_stats = {'train': {}, 'validation': {}, 'test': {}}
        
        for class_name in classes:
            print(f"\n📂 Processing class: {class_name}")
            
            class_dir = self.raw_data_path / class_name
            if not class_dir.exists():
                continue
            
            # Create class subdirectories in train/val/test
            for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
                (split_dir / class_name).mkdir(exist_ok=True)
            
            # Get all valid image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(class_dir.glob(ext)))
            
            if len(image_files) == 0:
                print(f"   No images found for {class_name}")
                continue
            
            # Split into train and temp (val+test)
            train_files, temp_files = train_test_split(
                image_files, 
                train_size=train_split, 
                random_state=42
            )
            
            # Split temp into val and test
            val_ratio = val_split / (val_split + (1 - train_split - val_split))
            val_files, test_files = train_test_split(
                temp_files, 
                train_size=val_ratio, 
                random_state=42
            )
            
            # Process and save images
            splits = {
                'train': (train_files, self.train_dir),
                'validation': (val_files, self.val_dir),
                'test': (test_files, self.test_dir)
            }
            
            for split_name, (files, output_dir) in splits.items():
                processed_count = 0
                class_output_dir = output_dir / class_name
                
                for i, image_file in enumerate(files):
                    output_file = class_output_dir / f"{class_name}_{i:04d}.jpg"
                    if self.resize_and_save_image(image_file, output_file):
                        processed_count += 1
                
                processed_stats[split_name][class_name] = processed_count
                print(f"   {split_name}: {processed_count} images")
        
        return processed_stats
    
    def generate_statistics(self, stats):
        """Generate and save dataset statistics."""
        print("\n📊 Generating dataset statistics...")
        
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Calculate totals
        splits = ['train', 'validation', 'test']
        class_totals = {}
        split_totals = {}
        
        for split in splits:
            split_totals[split] = sum(stats[split].values())
            
        for class_name in self.config['classes']:
            class_totals[class_name] = sum(stats[split].get(class_name, 0) for split in splits)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Samples per class
        classes = list(class_totals.keys())
        counts = list(class_totals.values())
        ax1.bar(classes, counts, color='skyblue')
        ax1.set_title('Samples per Class')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Images')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Split distribution
        split_names = list(split_totals.keys())
        split_counts = list(split_totals.values())
        ax2.pie(split_counts, labels=split_names, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Train/Validation/Test Split')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
        print(f"📈 Statistics plot saved to {output_dir}/dataset_statistics.png")
        
        # Save detailed statistics
        with open(output_dir / 'dataset_stats.txt', 'w') as f:
            f.write("InfraOwl Dataset Statistics\n")
            f.write("=" * 30 + "\n\n")
            
            f.write("Split Distribution:\n")
            for split, count in split_totals.items():
                f.write(f"{split.capitalize()}: {count} images\n")
            
            f.write("\nClass Distribution:\n")
            for class_name, count in class_totals.items():
                f.write(f"{class_name}: {count} images\n")
            
            f.write("\nDetailed Breakdown:\n")
            for split in splits:
                f.write(f"\n{split.capitalize()}:\n")
                for class_name in self.config['classes']:
                    count = stats[split].get(class_name, 0)
                    f.write(f"  {class_name}: {count}\n")
        
        print(f"📄 Detailed statistics saved to {output_dir}/dataset_stats.txt")
    
    def run(self):
        """Run the complete preprocessing pipeline."""
        print("🚀 Starting InfraOwl data preprocessing...")
        
        try:
            # Step 1: Validate raw data
            raw_stats = self.validate_raw_data()
            
            # Step 2: Process and split data
            processed_stats = self.split_and_process_data()
            
            # Step 3: Generate statistics
            self.generate_statistics(processed_stats)
            
            print("\n✅ Data preprocessing completed successfully!")
            print(f"📁 Processed data saved to: {self.processed_data_path}")
            
        except Exception as e:
            print(f"\n❌ Error during preprocessing: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Preprocess InfraOwl training data')
    parser.add_argument('--config', 
                       default='configs/training_config.yaml',
                       help='Path to training configuration file')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.config)
    preprocessor.run()


if __name__ == "__main__":
    main()