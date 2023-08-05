import PIL.Image

from med_palm.model import MedPalmTokenizer, MedPalm

# Instantiate tokenizer and model
tokenizer = MedPalmTokenizer()
model = MedPalm(num_tokens=1024)

# Load the image
image = PIL.Image.open('image-9.png')

# Create a list of different length text sequences for testing
texts = ["This is a short sequence."]

# Process each sequence and feed it into the model
for text in texts:
    sample = {"image": image, "target_text": text}
    tokenized_sample = tokenizer.tokenize(sample)
    
    text_tokens = tokenized_sample["text_tokens"]
    images = tokenized_sample["images"]

    # Feed tokenized data into the model
    output = model(text_tokens, images)
    
    print(f"Text: {text}")
    print(f"Number of tokens: {text_tokens.shape[1]}")
    print(f"Output size: {output}")
    print("\n-----\n")
