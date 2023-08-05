import PIL.Image

from med_palm.model import MedPalmTokenizer, MedPalm

# Instantiate tokenizer and model
tokenizer = MedPalmTokenizer()
model = MedPalm(num_tokens=50248)

# Load the image
image = PIL.Image.open('image-9.png')

# Create a list of different length text sequences for testing
texts = ["This is a short sequence.",
         "This is a longer sequence that contains more words and thus more tokens when processed by the tokenizer.",
         "This is an even longer sequence. It contains even more words and will thus result in even more tokens when processed by the tokenizer. The purpose of this is to test how the model handles sequences of varying lengths.",
         "This is the longest sequence used in this test. It is significantly longer than the other sequences and will result in a significantly higher number of tokens when processed by the tokenizer. The purpose of this sequence is to test the upper limits of the model's capacity to handle long sequences."]

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
