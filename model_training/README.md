# InfraOwl Model Training Pipeline

Complete machine learning pipeline for training custom infrastructure detection models for the InfraOwl Flutter application.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Organize Your Data**
   ```
   data/raw_images/
   ├── pothole/          # Add pothole images here
   ├── crack/            # Add crack images here
   ├── broken_pipe/      # Add broken pipe images here
   ├── damaged_road/     # Add damaged road images here
   └── normal/           # Add normal infrastructure images here
   ```

3. **Run the Pipeline**
   ```bash
   # Step 1: Preprocess data
   python scripts/data_preprocessing.py
   
   # Step 2: Train model
   python scripts/train_model.py
   
   # Step 3: Convert to mobile format
   python scripts/convert_to_tflite.py
   
   # Step 4: Evaluate performance
   python scripts/evaluate_model.py
   ```

## 📁 Directory Structure

```
model_training/
├── configs/
│   └── training_config.yaml      # Training configuration
├── data/
│   ├── raw_images/              # Original images by class
│   ├── processed/               # Processed train/val/test splits
│   └── labels.txt               # Class labels
├── scripts/
│   ├── data_preprocessing.py    # Data preparation
│   ├── train_model.py          # Model training
│   ├── convert_to_tflite.py    # Mobile conversion
│   └── evaluate_model.py       # Model evaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Interactive data analysis
│   ├── 02_data_preprocessing.ipynb  # Data preparation
│   └── 03_model_training.ipynb      # Training workflow
├── models/
│   ├── saved_models/            # Trained Keras models
│   ├── checkpoints/             # Training checkpoints
│   └── tflite_models/           # Mobile-optimized models
└── outputs/                     # Generated reports and plots
```

## 🎯 Training Configuration

Edit `configs/training_config.yaml` to customize:

### Model Settings
- **Architecture**: `efficientnet_lite0`, `mobilenetv3_small`
- **Input Size**: Default `[224, 224, 3]`
- **Classes**: Update for your specific infrastructure issues

### Training Parameters
- **Epochs**: Number of training iterations
- **Batch Size**: Samples per training step
- **Learning Rate**: Initial learning rate
- **Optimizer**: `adam`, `sgd`

### Data Augmentation
- **Rotation**: Random rotation range
- **Shifts**: Width/height shift ranges
- **Zoom**: Random zoom range
- **Brightness**: Brightness variation range

## 📊 Jupyter Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Analyze dataset structure and statistics
- Visualize sample images from each class
- Identify data quality issues
- Generate recommendations

### 2. Data Preprocessing (`02_data_preprocessing.ipynb`)
- Interactive data preprocessing
- Visualize train/validation/test splits
- Preview data augmentation
- Generate dataset statistics

### 3. Model Training (`03_model_training.ipynb`)
- Complete training workflow
- Real-time training visualization
- Model evaluation and testing
- TensorFlow Lite conversion

## 🛠️ Scripts

### Data Preprocessing
```bash
python scripts/data_preprocessing.py --config configs/training_config.yaml
```
- Validates raw data structure
- Resizes images to target dimensions
- Splits data into train/validation/test sets
- Generates preprocessing statistics

### Model Training
```bash
python scripts/train_model.py --config configs/training_config.yaml
```
- Creates model architecture (transfer learning)
- Trains with data augmentation
- Implements callbacks (early stopping, checkpoints)
- Generates training visualizations

### TensorFlow Lite Conversion
```bash
python scripts/convert_to_tflite.py --config configs/training_config.yaml
```
- Converts trained models to TFLite format
- Applies quantization for smaller models
- Benchmarks inference speed
- Copies optimized model to Flutter assets

### Model Evaluation
```bash
python scripts/evaluate_model.py --config configs/training_config.yaml
```
- Comprehensive model evaluation
- Confusion matrix and classification reports
- Compares Keras vs TFLite performance
- Generates evaluation visualizations

## 🏗️ Model Architecture

### Transfer Learning Approach
1. **Base Model**: Pre-trained EfficientNet-Lite or MobileNet
2. **Feature Extraction**: Frozen pre-trained layers
3. **Custom Head**: Dense layers for infrastructure classification
4. **Fine-tuning**: Optional unfreezing for domain adaptation

### Supported Architectures
- **EfficientNet-Lite0**: Balanced accuracy and efficiency
- **MobileNetV3-Small**: Ultra-lightweight for mobile
- **Custom CNN**: Build from scratch if needed

## 📱 Mobile Deployment

### TensorFlow Lite Optimization
- **Dynamic Quantization**: Reduces model size
- **INT8 Quantization**: Further compression with representative dataset
- **Float16 Quantization**: Balance between size and accuracy

### Integration with Flutter
1. **Model File**: Copy `.tflite` file to `assets/` folder
2. **Labels**: Update `labels.txt` with your classes
3. **Service Update**: Modify `ModelService` if needed

## 📈 Performance Monitoring

### Training Metrics
- **Accuracy**: Classification accuracy
- **Loss**: Training and validation loss
- **Learning Rate**: Adaptive learning rate schedule

### Evaluation Metrics
- **Confusion Matrix**: Class-wise prediction accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Balanced performance metric

### TensorBoard Integration
```bash
tensorboard --logdir logs/
```
- Real-time training monitoring
- Loss and accuracy curves
- Model architecture visualization
- Hyperparameter comparison

## 🎯 Best Practices

### Data Collection
- **Minimum**: 100+ images per class
- **Quality**: High-resolution, well-lit images
- **Variety**: Different angles, conditions, scenarios
- **Balance**: Similar number of samples per class

### Training Tips
- **Start Small**: Begin with fewer epochs for quick iteration
- **Monitor Overfitting**: Watch validation vs training metrics
- **Data Augmentation**: Essential for small datasets
- **Transfer Learning**: Faster convergence than training from scratch

### Model Optimization
- **Early Stopping**: Prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rate
- **Model Checkpoints**: Save best performing models
- **Quantization**: Optimize for mobile deployment

## 🔧 Troubleshooting

### Common Issues

**Low Accuracy**
- Increase training epochs
- Add more training data
- Adjust learning rate
- Enable/tune data augmentation

**Overfitting**
- Add dropout layers
- Reduce model complexity
- Increase validation data
- Enable early stopping

**Slow Training**
- Reduce batch size
- Use GPU acceleration
- Optimize data loading
- Reduce image resolution

**Large Model Size**
- Use smaller architecture
- Apply quantization
- Prune unnecessary layers
- Optimize for mobile

### GPU Setup
```python
# Check GPU availability
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## 📚 Additional Resources

### Documentation
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [Transfer Learning Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)

### Model Architectures
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [MobileNet V3 Paper](https://arxiv.org/abs/1905.02244)
- [Transfer Learning Best Practices](https://arxiv.org/abs/1911.02685)

## 🤝 Contributing

1. **Add New Architectures**: Extend `train_model.py`
2. **Improve Preprocessing**: Enhance `data_preprocessing.py`
3. **Add Metrics**: Extend evaluation scripts
4. **Documentation**: Update notebooks and README

## 📄 License

This model training pipeline is part of the InfraOwl project. See the main project license for details.