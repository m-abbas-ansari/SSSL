from pathlib import Path

class args:
    seed = 41
    epochs=1000
    batch_size= 32
    img_height = 224
    img_width = 224
    ffn_size = 512
    hidden_size = ffn_size*4
    workers= 1
    learning_rate_weights=0.2
    learning_rate_biases=0.0048
    weight_decay=1e-6
    lambd=0.0051
    projector=[hidden_size*4]*3
    print_freq=100
    checkpoint_dir=Path('../checkpoint/')
    data_dir = "../../input/fixation-datasets/FIXATION_DATASET"
    datasets = ['SALICON', 'EMOD', 'FIWI', 'MITLowRes', 'OSIE', 'SIENA12', 'TORONTO', 'VIU']
    transform = dict(
        img_size = (img_height, img_width),
        noise = 0.6,
        drop = 0.4,
        reversal = 0.5,
        rotation = 0.5,
    )