import argparse
import torch

from models import build_model
# from .detr import build_model
# from .detr import build
def get_args():
    parser = argparse.ArgumentParser('DETR Detector', add_help=False)
    # Training hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float, help="Learning rate")
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help="Backbone learning rate")
    parser.add_argument('--batch_size', default=2, type=int, help="Batch size")
    parser.add_argument('--device', default='cpu', type=str, help="Device to use")

    # Backbone and dataset
    parser.add_argument('--backbone', default='resnet18', type=str, help="CNN backbone type")
    parser.add_argument('--dataset_file', default='coco', type=str, help="Dataset identifier (e.g., coco)")

    # DETR model hyperparameters
    parser.add_argument('--num_queries', default=100, type=int, help="Number of object queries")
    parser.add_argument('--aux_loss', action='store_true', help="Enable auxiliary decoding losses")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Transformer hidden dimension")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout rate")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Feedforward network dimension")
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoder layers")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoder layers")
    parser.add_argument('--pre_norm', action='store_true', help="Apply layer normalization before operations")

    # For segmentation (if you want instance segmentation)
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if true")

    # Matching and loss coefficients
    parser.add_argument('--set_cost_class', default=1, type=float, help="Classification cost weight")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="BBox L1 cost weight")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="GIoU cost weight")
    parser.add_argument('--bbox_loss_coef', default=5, type=float, help="BBox loss coefficient")
    parser.add_argument('--giou_loss_coef', default=2, type=float, help="GIoU loss coefficient")
    parser.add_argument('--mask_loss_coef', default=1, type=float, help="Mask loss coefficient")
    parser.add_argument('--dice_loss_coef', default=1, type=float, help="Dice loss coefficient")
    parser.add_argument('--eos_coef', default=0.1, type=float, help="No-object (end of sequence) class weight")

    # Positional encoding and other settings
    parser.add_argument('--position_embedding', default='sine', choices=['sine', 'learned'], help="Type of positional encoding")
    parser.add_argument('--dilation', action='store_true', help="Use dilation in the backbone")

    args = parser.parse_args()
    return args
def count_module_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

if __name__ == '__main__':
    args = get_args()
    # Build the DETR model, criterion (loss), and postprocessors
    model, criterion, postprocessors = build_model(args)
    model.to(args.device)

    # Print model architecture
    print(model)

    # Breakdown by major component
    backbone_params = count_module_params(model.backbone)
    transformer_params = count_module_params(model.transformer)
    class_embed_params = count_module_params(model.class_embed)
    bbox_embed_params = count_module_params(model.bbox_embed)
    query_embed_params = count_module_params(model.query_embed)
    input_proj_params = count_module_params(model.input_proj)

    total_params = count_module_params(model)

    # Display the parameter counts
    print("DETR Parameter Breakdown:")
    print(f"Backbone:              {backbone_params:,}")
    print(f"Transformer:           {transformer_params:,}")
    print(f"Input Projection:      {input_proj_params:,}")
    print(f"Query Embedding:       {query_embed_params:,}")
    print(f"Classification Head:   {class_embed_params:,}")
    print(f"Bounding Box Head:     {bbox_embed_params:,}")
    print(f"---------------------------------------")
    print(f"Total Parameters:      {total_params:,}")
    # Here you would add your dataset loader and training loop.
    # For each batch:
    #   1. Forward pass: outputs = model(samples)
    #   2. Compute losses: loss_dict = criterion(outputs, targets)
    #   3. Backpropagation and optimizer step.
