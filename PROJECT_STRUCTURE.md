# Mask2Former Project Structure

## Directory Layout

```
E:\soft\code\Mask2former\
├── scripts/                    # Python scripts
│   ├── analyze_masks.py       # Mask analysis & visualization
│   ├── convert_masks.py       # Convert mask pixel values (255→0)
│   ├── example.py             # Example usage with pretrained model
│   ├── finetune_custom.py     # Fine-tuning script
│   └── inference.py           # Inference with fine-tuned model
│
├── train/                      # Training data
│   ├── images_png/            # Training images (PNG format)
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   ├── masks_png/             # Original masks (with pixel value 255)
│   │   ├── 000001.png
│   │   ├── 000002.png
│   │   └── ...
│   └── masks_converted/       # Converted masks (255→0)
│       ├── 000001.png
│       ├── 000002.png
│       └── ...
│
├── results/                    # Output results
│   ├── models/                # Saved models
│   │   └── finetuned_model/   # Fine-tuned model
│   │       ├── config.json
│   │       ├── preprocessor_config.json
│   │       └── pytorch_model.bin
│   └── visualizations/        # Visualization results
│       ├── Mask_Analysis_*.png
│       ├── inference_result.png
│       └── segmentation_result.png
│
├── models--facebook--mask2former-swin-tiny-coco-panoptic/  # Pretrained model cache
│
├── config.json                 # Model configuration
├── download.py                 # Download pretrained model
├── prepare_data.py            # Prepare training data
├── preprocessor_config.json   # Preprocessor configuration
├── pytorch_model.bin          # Pretrained model weights
└── sample_image.jpg           # Sample image for testing
```

## Quick Start

### 1. Analyze Masks
```bash
cd E:\soft\code\Mask2former
python scripts/analyze_masks.py
```

### 2. Convert Masks (255→0)
```bash
python scripts/convert_masks.py
```

### 3. Fine-tune Model
```bash
python scripts/finetune_custom.py
```

### 4. Run Inference
```bash
python scripts/inference.py
```

## Mask Pixel Values

| Pixel Value | Color | Description |
|-------------|-------|-------------|
| 0 | Black | Background |
| 1 | Red | Class 1 (Hole) |
| 2 | Green | Class 2 (Slot) |
| 3 | Blue | Class 3 |

## Training Configuration

- **Model**: Mask2Former (facebook/mask2former-swin-tiny-coco-panoptic)
- **Device**: CPU (can switch to CUDA if available)
- **Epochs**: 5
- **Batch Size**: 2
- **Learning Rate**: 5e-5
- **Image Size**: 512x512

## Notes

1. Original masks contain pixel value 255 (ignore region)
2. Converted masks replace 255 with 0 (background)
3. Use `train/masks_converted/` for training
4. Results are saved in `results/` directory