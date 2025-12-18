import math
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class GutenbergDataset(Dataset):
    def __init__(self, text, char2idx, seq_length=40, step=3):
        """
        Custom Dataset to create overlapping sequences.
        """
        self.char2idx = char2idx
        self.seq_length = seq_length

        # 1. Encode the entire corpus into integers
        print("Encoding corpus...")
        self.encoded_text = [char2idx[c] for c in text]
        self.total_chars = len(self.encoded_text)

        # 2. Pre-calculate the starting indices for our sliding window
        # We stop early enough so the last sequence has a target
        self.indices = list(range(0, self.total_chars - seq_length, step))

        print(f"Total Sequences Created: {len(self.indices):,}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the start index for this sequence
        start = self.indices[idx]
        end = start + self.seq_length

        # Input sequence: characters at positions [t, t+1, ... t+39]
        input_chunk = self.encoded_text[start:end]

        # Target sequence: characters at positions [t+1, t+2, ... t+40]
        target_chunk = self.encoded_text[start + 1 : end + 1]

        # Convert to PyTorch LongTensors (integers)
        return torch.tensor(input_chunk, dtype=torch.long), torch.tensor(
            target_chunk, dtype=torch.long
        )


class LSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        # 1. Embedding Layer
        # Converts integer indices to dense vectors.
        # Ideally acts like a "learnable" One-Hot encoding.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2. LSTM Layers
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=0.3,
            batch_first=True,
        )

        # 3. Fully Connected (Output) Layer
        # Maps the LSTM hidden states back to the vocabulary size (scores for each character)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Learning Rate
        self.lr = lr

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)

        # Step 1: Embed
        # out shape: (batch_size, seq_length, embed_dim)
        out = self.embedding(x)

        # Step 2: LSTM
        # out shape: (batch_size, seq_length, hidden_dim)
        # hidden contains the (h_n, c_n) states for the next sequence
        out, hidden = self.lstm(out, hidden)

        # Step 3: Decode
        # We process the entire sequence at once (Many-to-Many architecture)
        out = self.fc(out)
        # final out shape: (batch_size, seq_length, vocab_size)

        return out, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass (ignore hidden states during training as we reset per batch)
        y_hat, _ = self.forward(x)

        # Reshape for CrossEntropyLoss
        # Target y is (batch, seq), Prediction y_hat is (batch, seq, vocab)
        # PyTorch Loss expects: (N, C) and (N)
        # We flatten batch and sequence dimensions
        loss = F.cross_entropy(y_hat.view(-1, self.hparams.vocab_size), y.view(-1))

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def generate_text(
    model, char2idx, idx2char, start_string, generation_length=200, temperature=1.0
):
    """
    Generates text using the trained LSTM.

    Args:
        start_string (str): The seed text to start the generation.
        generation_length (int): How many characters to generate.
        temperature (float): Controls randomness.
                             - Low (0.2): Conservative, repetitive, safe.
                             - High (1.0): Creative, diverse, prone to errors.
    """
    model.eval()  # Switch to evaluation mode

    # 1. Prepare the seed
    # Convert string to indices
    input_indices = [char2idx[c] for c in start_string]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(
        0
    )  # Add batch dim -> (1, seq_len)

    generated_text = start_string

    # Initialize hidden state
    hidden = None

    with torch.no_grad():
        for _ in range(generation_length):
            # 2. Forward pass
            output, hidden = model(input_tensor, hidden)

            # 3. Get the prediction for the last character
            # output shape: (1, seq_len, vocab_size)
            # We only care about the logits for the very last time step
            last_logits = output[0, -1, :]

            # 4. Apply Temperature
            # Higher temp makes the distribution flatter (more random)
            # Lower temp makes peaks sharper (more deterministic)
            scaled_logits = last_logits / temperature
            probs = F.softmax(scaled_logits, dim=0)

            # 5. Sample from the distribution
            # We don't just pick the 'argmax' (highest prob), we sample based on probability
            # This allows for variety.
            predicted_index = torch.multinomial(probs, 1).item()

            # 6. Append to result
            predicted_char = idx2char[predicted_index]
            generated_text += predicted_char

            # 7. Update input for next step
            # We feed the NEW character back in as the next input
            input_tensor = torch.tensor([[predicted_index]], dtype=torch.long)

    return generated_text


def calculate_quantitative_metrics(model, data_loader):
    """
    Calculates Perplexity and Bits-Per-Character (BPC) on the validation/training set.
    """
    model.eval()
    total_loss = 0
    total_batches = 0

    # Standard CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss()

    print("Calculating Loss Metrics...")
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch

            # Forward pass
            # y_hat shape: (Batch, Seq, Vocab)
            y_hat, _ = model(x)

            # Reshape for loss calculation
            # Flatten batch and sequence dims
            loss = criterion(y_hat.view(-1, model.hparams.vocab_size), y.view(-1))

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches

    # 1. Perplexity (PPL) = exp(Loss)
    # Measures how "surprised" the model is. Lower is better.
    perplexity = math.exp(avg_loss)

    # 2. Bits Per Character (BPC) = Loss / ln(2)
    # Standard metric for character-level models.
    bpc = avg_loss / math.log(2)

    return avg_loss, perplexity, bpc


def calculate_spelling_accuracy(
    model, char2idx, idx2char, ref_text, num_samples=5, gen_len=200
):
    """
    Generates text and checks what percentage of words are valid
    (exist in the original Jane Austen corpus).
    """
    # 1. Build a 'Dictionary' from the original text
    # Remove punctuation to isolate words
    translator = str.maketrans("", "", string.punctuation)
    clean_corpus = ref_text.translate(translator).lower()
    valid_vocabulary = set(clean_corpus.split())

    print(f"\nReference Vocabulary Size: {len(valid_vocabulary)} unique words")

    total_words = 0
    correct_words = 0

    print(f"Generating {num_samples} samples to check spelling...")

    for _ in range(num_samples):
        # Pick random seed
        start = np.random.randint(0, len(ref_text) - 100)
        seed = ref_text[start : start + 40]

        # Generate with Temperature 0.5 (Fair balance)
        gen_text = generate_text(
            model, char2idx, idx2char, seed, generation_length=gen_len, temperature=0.5
        )

        # Clean generated text (ignore the seed part)
        generated_content = gen_text[40:]
        clean_gen = generated_content.translate(translator).lower().split()

        for word in clean_gen:
            # We ignore very short fragments which might be artifacts of splitting
            if len(word) > 1:
                total_words += 1
                if word in valid_vocabulary:
                    correct_words += 1

    accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
    return accuracy
