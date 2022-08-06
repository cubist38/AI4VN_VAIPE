from .swin_transformer import swin_base_transformer_384, swin_tiny_transformer, swin_large_transformer

swin_transformer_map = {
    'swin_tiny_transformer': swin_tiny_transformer,
    'swin_base_transformer_384': swin_base_transformer_384,
    'swin_large_transformer': swin_large_transformer
}