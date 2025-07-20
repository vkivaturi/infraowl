# Data Organization Guide

## Directory Structure

```
data/
├── raw_images/           # Original images organized by class
│   ├── pothole/         # Images of potholes
│   ├── crack/           # Images of road/infrastructure cracks
│   ├── broken_pipe/     # Images of broken pipes/water issues
│   ├── damaged_road/    # Images of damaged road surfaces
│   └── normal/          # Images of normal/undamaged infrastructure
├── processed/           # Processed and split data
│   ├── train/          # Training data (70%)
│   ├── validation/     # Validation data (20%)
│   └── test/           # Test data (10%)
└── labels.txt          # Class labels file
```

## Adding Your Images

### Step 1: Organize Raw Images
1. Place your images in the appropriate class folders under `raw_images/`
2. Supported formats: .jpg, .jpeg, .png
3. Recommended: At least 100-200 images per class for good performance

### Step 2: Image Quality Guidelines
- **Resolution**: Minimum 224x224 pixels (will be resized automatically)
- **Quality**: Clear, well-lit images work best
- **Variety**: Include different angles, lighting conditions, and scenarios
- **Consistency**: Try to maintain similar framing and distance

### Step 3: Labeling Guidelines

#### Pothole Class:
- Clear depressions or holes in road surface
- Various sizes from small to large potholes
- Different road surfaces (asphalt, concrete)

#### Crack Class:
- Linear cracks in roads, sidewalks, or structures
- Include hairline cracks to major fractures
- Both filled and unfilled cracks

#### Broken Pipe Class:
- Visible pipe damage or leaks
- Water pooling from underground sources
- Exposed or damaged utility lines

#### Damaged Road Class:
- General road surface deterioration
- Worn surfaces, missing chunks
- Edge deterioration or shoulder damage

#### Normal Class:
- Well-maintained infrastructure
- Clean, undamaged surfaces
- Good condition roads and structures

## Data Processing

After organizing your raw images, run the preprocessing script:

```bash
python scripts/data_preprocessing.py
```

This will:
- Resize images to 224x224 pixels
- Split data into train/validation/test sets
- Apply data validation checks
- Generate data statistics

## Data Statistics

After processing, check the generated statistics:
- Class distribution
- Image quality metrics
- Dataset balance recommendations

## Tips for Better Results

1. **Balanced Dataset**: Try to have similar number of images per class
2. **Data Augmentation**: The training pipeline includes augmentation for better generalization
3. **Quality over Quantity**: 100 high-quality images per class is better than 500 poor quality ones
4. **Regular Updates**: Continuously add new images to improve model performance