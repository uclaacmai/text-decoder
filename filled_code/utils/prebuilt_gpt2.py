import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
from transformers import LogitsProcessorList, MinLengthLogitsProcessor

# Load pretrained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # can replace this tokenizer with a customized one

# Load pretrained GPT-2 model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define a new embedding layer
new_embedding_dim = 768  # Change this to your desired embedding dimension
new_embedding_layer = torch.nn.Embedding(tokenizer.vocab_size, new_embedding_dim)

# can train this new embedding layer
#model.transformer.wte = new_embedding_layer

# Set the model to evaluation mode
model.eval()

# Encode input text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
with torch.no_grad():
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)


def bigram_attention_mask(seq_length):
    mask = torch.tril(torch.ones(seq_length, seq_length)).unsqueeze(0).unsqueeze(0)
    mask = mask - torch.tril(torch.ones(seq_length, seq_length), diagonal=-2).unsqueeze(0).unsqueeze(0)
    mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_with_bigram_attention(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Prepare the initial input tensor
    outputs = input_ids
    for _ in range(max_length - len(input_ids[0])):
        # Generate bigram attention mask for the current length
        attn_mask = bigram_attention_mask(outputs.size(1))

        # Pass through the model
        with torch.no_grad():
            logits = model(input_ids=outputs, attention_mask=attn_mask).logits

        # Get the next token
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        # Append the next token to the outputs
        outputs = torch.cat((outputs, next_token), dim=1)

    return outputs

# Generate text
input_text = "Once upon a time"
output_ids = generate_with_bigram_attention(model, tokenizer, input_text, max_length=20)

# Decode the generated text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(generated_text)


