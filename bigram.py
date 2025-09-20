import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------
# Hyperparameters (config)
# ---------------------------
BATCH_SIZE = 32  # câte secvențe procesăm în paralel
BLOCK_SIZE = (
    8  # cât de mult context (cuvinte/tokens) citim înainte să prezicem următorul
)
MAX_ITERS = 3000
EVAL_INTERVAL = 300  # la câți pași facem evaluare pe datele de validare
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200  # câte batchuri evaluăm ca să facem media pierderii (loss)

# Setăm seed-ul ca să avem rezultate reproducibile
torch.manual_seed(1337)

# ---------------------------
# Încărcăm și procesăm datele
# ---------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Extragem toate caracterele unice din text și creăm un vocabular
unique_chars = sorted(list(set(text)))
VOCAB_SIZE = len(unique_chars)

# Dicționare pentru a converti caracter <-> index
char_to_index = {ch: i for i, ch in enumerate(unique_chars)}
index_to_char = {i: ch for i, ch in enumerate(unique_chars)}

# Functii pentru encoding/decoding (string -> listă de indici și invers)
encode = lambda s: [char_to_index[c] for c in s]
decode = lambda l: "".join([index_to_char[i] for i in l])

# Convertim tot textul într-un tensor de indici
data = torch.tensor(encode(text), dtype=torch.long)

# Împărțim datele: 90% pentru antrenament, 10% pentru validare
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]


# ---------------------------
# Funcție pentru a genera batchuri
# ---------------------------
def get_batch(split):
    dataset = train_data if split == "train" else val_data
    # Alegem random puncte de start pentru fiecare secvență
    start_indices = torch.randint(len(dataset) - BLOCK_SIZE, (BATCH_SIZE,))

    # Construim inputul x și ținta y (care este inputul shift-uit la dreapta cu 1)
    input_batch = torch.stack([dataset[i : i + BLOCK_SIZE] for i in start_indices])
    target_batch = torch.stack(
        [dataset[i + 1 : i + BLOCK_SIZE + 1] for i in start_indices]
    )
    return input_batch.to(DEVICE), target_batch.to(DEVICE)


# ---------------------------
# Estimăm loss-ul mediu pe train și val (fără gradient)
# ---------------------------
@torch.no_grad()
def estimate_loss():
    loss_results = {}
    model.eval()  # dezactivăm dropout etc.
    for split in ["train", "val"]:
        split_losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            split_losses[k] = loss.item()
        loss_results[split] = split_losses.mean()
    model.train()  # revenim la mod de training
    return loss_results


# ---------------------------
# Modelul propriu-zis
# ---------------------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Fiecare token (caracter) va avea un rând în această matrice care spune ce scoruri (logits) să dea altor tokeni
        self.token_embedding_lookup = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_tokens, target_tokens=None):
        # Primim un tensor de indici: formă (batch, time)
        # Îi mapăm direct la logits: scoruri brute pt. fiecare token următor posibil
        logits = self.token_embedding_lookup(input_tokens)  # Formă: (B, T, C)

        if target_tokens is None:
            return logits, None  # Dacă nu avem ținte, returnăm doar predicțiile

        # Altfel, calculăm cross-entropy loss
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = target_tokens.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, context_tokens, max_new_tokens):
        # Pornim de la un context (ex: un caracter) și generăm max_new_tokens caractere noi
        for _ in range(max_new_tokens):
            logits, _ = self(context_tokens)
            last_logits = logits[:, -1, :]  # ne uităm doar la ultimul pas din secvență
            probs = F.softmax(
                last_logits, dim=-1
            )  # transformăm logits în probabilități
            next_token = torch.multinomial(
                probs, num_samples=1
            )  # alegem un token aleator conform probabilităților
            context_tokens = torch.cat(
                (context_tokens, next_token), dim=1
            )  # adăugăm tokenul generat la secvență
        return context_tokens


# ---------------------------
# Inițializăm modelul și optimizerul
# ---------------------------
model = BigramLanguageModel(VOCAB_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ---------------------------
# Loop-ul de antrenament
# ---------------------------
for iteration in range(MAX_ITERS):
    if iteration % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(
            f"Step {iteration}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}"
        )

    x_batch, y_batch = get_batch("train")
    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ---------------------------
# Generăm text nou după antrenament
# ---------------------------
initial_context = torch.zeros(
    (1, 1), dtype=torch.long, device=DEVICE
)  # începem cu caracterul 0 (de obicei \n)
output_sequence = model.generate(initial_context, max_new_tokens=500)
print(decode(output_sequence[0].tolist()))
