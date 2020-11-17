import os
import torch
from model import LitAutoEncoder

# ----------------------------------
# to use as embedding extractor
# ----------------------------------
autoencoder = LitAutoEncoder.load_from_checkpoint(
        './lightning_logs/version_0/checkpoints/best.ckpt')
autoencoder.eval()
output = autoencoder(torch.rand(1, 28 * 28))

# ----------------------------------
# to use as embedding extractor
# ----------------------------------
encoder_model = autoencoder.encoder
encoder_model.eval()
output = encoder_model(torch.rand(64, 3))

# ----------------------------------
# to use as image generator
# ----------------------------------
decoder_model = autoencoder.decoder
decoder_model.eval()
output = decoder_model(torch.rand(1, 28 * 28))

# ----------------------------------
# torchscript
# ----------------------------------
autoencoder = LitAutoEncoder()
torch.jit.save(autoencoder.to_torchscript(), "model.pt")
os.path.isfile("model.pt")

# ----------------------------------
# onnx
# ----------------------------------
# with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
#      autoencoder = LitAutoEncoder()
#      input_sample = torch.randn((1, 28 * 28))
#      autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
#      os.path.isfile(tmpfile.name)
