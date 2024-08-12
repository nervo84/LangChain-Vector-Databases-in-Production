from transformers import AutoTokenizer

# Download and load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print( tokenizer.vocab )

token_ids = tokenizer.encode("Questo è un testo di esempio per testare il tokenizzatore.")

print( "Tokens:   ", tokenizer.convert_ids_to_tokens( token_ids ) )
print( "Token IDs:", token_ids )