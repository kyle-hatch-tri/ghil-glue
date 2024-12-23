from susie.model import load_vae, load_text_encoder
from transformers import T5Tokenizer, T5EncoderModel

pretrained_path = "runwayml/stable-diffusion-v1-5:flax"
vae_encode, vae_decode = load_vae(pretrained_path)
tokenize, untokenize, text_encode = load_text_encoder(pretrained_path)


tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
text_model = T5EncoderModel.from_pretrained("google/flan-t5-base")

print("Done downloading models.")