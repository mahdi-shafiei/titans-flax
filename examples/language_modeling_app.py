"""Language modeling application with Titans using real text data.

This example demonstrates:
1. Text preprocessing and tokenization using tiktoken
2. Training a Titans model on Shakespeare's sonnets
3. Text generation with neural memory adaptation
4. Model evaluation and memory analysis
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import optax
import tiktoken

from titans.config import TitansConfig, create_config_for_task
from titans.core import TitansModel
from titans.serialization import TitansCheckpoint


class TiktokenTokenizer:
    """Proper tokenizer using tiktoken BPE encoding."""

    def __init__(self, encoding_name: str = "gpt2", _vocab_size: int = 50257):
        """Initialize tokenizer with tiktoken encoding.

        Args:
            encoding_name: Name of tiktoken encoding (gpt2, cl100k_base, etc.)
            _vocab_size: Vocabulary size (for compatibility, actual vocab size
                       determined by encoding)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab

        # For GPT-2 encoding, use standard special tokens
        if encoding_name == "gpt2":
            # GPT-2 uses 50256 as the endoftext token
            try:
                self.eos_token_id = self.encoding.encode(
                    "<|endoftext|>", allowed_special={"<|endoftext|>"}
                )[0]
            except (ValueError, IndexError):
                self.eos_token_id = 50256

            self.pad_token_id = 50256  # Same as EOS for simplicity
            self.bos_token_id = 50256  # Same as EOS for simplicity
        else:
            # Default fallback tokens
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.bos_token_id = 2

    def encode(self, text: str, max_length: int = 512) -> list[int]:
        """Encode text to token IDs using tiktoken.

        Args:
            text: Input text to encode
            max_length: Maximum sequence length

        Returns:
            List of token IDs
        """
        # Encode with tiktoken
        tokens = self.encoding.encode(text)

        # Truncate if too long (leave space for BOS/EOS)
        if len(tokens) > max_length - 2:
            tokens = tokens[: max_length - 2]

        # Add BOS and EOS tokens
        return [self.bos_token_id, *tokens, self.eos_token_id]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text using tiktoken.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        # Filter out special tokens and padding
        filtered_tokens: list[int] = []
        for token_id in token_ids:
            if token_id == self.pad_token_id:
                continue
            if token_id == self.eos_token_id:
                break
            if token_id == self.bos_token_id:
                continue
            # Only include tokens within vocabulary range
            if 0 <= token_id < self.vocab_size:
                filtered_tokens.append(token_id)

        try:
            return self.encoding.decode(filtered_tokens)
        except Exception:
            # Fallback for invalid token sequences
            fallback_chars = [
                chr(65 + (token_id % 26)) for token_id in filtered_tokens[:10]
            ]
            return "".join(fallback_chars)

    def create_attention_mask(self, token_ids: list[int]) -> list[bool]:
        """Create attention mask for padding."""
        return [token_id != self.pad_token_id for token_id in token_ids]


class LanguageModelDataset:
    """Dataset for language modeling tasks."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: TiktokenTokenizer,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        """Initialize dataset.

        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            batch_size: Batch size
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        # Preprocess texts
        self.token_sequences: list[list[int]] = []
        for text in texts:
            tokens = tokenizer.encode(text, max_length)
            if len(tokens) >= 2:  # Must have at least input and target
                self.token_sequences.append(tokens)

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over batches."""
        for i in range(0, len(self.token_sequences), self.batch_size):
            batch_sequences = self.token_sequences[i : i + self.batch_size]

            # Pad sequences to same length
            max_len: int = min(
                max(len(seq) for seq in batch_sequences), self.max_length
            )

            input_ids: list[list[int]] = []
            target_ids: list[list[int]] = []
            attention_masks: list[list[int]] = []

            for seq in batch_sequences:
                # Truncate or pad
                # Create new variable to avoid overwriting loop var
                processed_seq: list[int] = seq
                if len(processed_seq) > max_len:
                    processed_seq = processed_seq[:max_len]
                else:
                    pad_tokens = [self.tokenizer.pad_token_id] * (
                        max_len - len(processed_seq)
                    )
                    processed_seq = processed_seq + pad_tokens

                # Create input (all tokens except last) and target (all tokens except first)
                input_seq: list[int] = processed_seq[:-1]
                target_seq: list[int] = processed_seq[1:]

                # Pad to consistent length
                if len(input_seq) < max_len - 1:
                    padding_length: int = max_len - 1 - len(input_seq)
                    input_seq = (
                        input_seq + [self.tokenizer.pad_token_id] * padding_length
                    )
                    target_seq = (
                        target_seq + [self.tokenizer.pad_token_id] * padding_length
                    )

                input_ids.append(input_seq)
                target_ids.append(target_seq)

                # Create attention mask
                mask: list[int] = [
                    1 if token_id != self.tokenizer.pad_token_id else 0
                    for token_id in input_seq
                ]
                attention_masks.append(mask)

            yield {
                "input_ids": jnp.array(input_ids, dtype=jnp.int32),
                "target_ids": jnp.array(target_ids, dtype=jnp.int32),
                "attention_mask": jnp.array(attention_masks, dtype=jnp.bool_),
            }


class TitansLanguageModel(nnx.Module):
    """Complete language model using Titans architecture."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        titans_config: TitansConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize language model.

        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            titans_config: Titans configuration
            rngs: Random number generators
            dtype: Data type
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding
        self.token_embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=d_model,
            rngs=rngs,
            dtype=dtype,
        )

        # Positional embedding
        max_position = 2048
        self.positional_embedding = nnx.Embed(
            num_embeddings=max_position,
            features=d_model,
            rngs=rngs,
            dtype=dtype,
        )

        # Titans backbone
        self.titans_model = TitansModel(
            variant=titans_config.variant,
            d_model=d_model,
            num_heads=titans_config.attention.num_heads,
            memory_depth=titans_config.memory.memory_depth,
            num_persistent_tokens=titans_config.memory.num_persistent_tokens,
            chunk_size=titans_config.architecture.chunk_size,
            window_size=titans_config.attention.window_size or 256,
            dropout_rate=titans_config.attention.dropout_rate,
            rngs=rngs,
            dtype=dtype,
        )

        # Output projection
        self.output_projection = nnx.Linear(
            in_features=d_model,
            out_features=vocab_size,
            use_bias=False,
            rngs=rngs,
            dtype=dtype,
        )

        # Layer normalization
        self.final_norm = nnx.LayerNorm(
            num_features=d_model,
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        deterministic: bool = False,
        reset_memory: bool = False,
    ) -> jax.Array:
        """Forward pass of language model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len)
            deterministic: Whether to disable memory updates and dropout
            reset_memory: Whether to reset Titans memory

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Positional embeddings
        positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        pos_embeds = self.positional_embedding(positions)

        # Combine embeddings
        embeddings = token_embeds + pos_embeds

        # Apply Titans model
        if reset_memory:
            self.titans_model.reset_memory()

        hidden_states = self.titans_model(
            embeddings,
            deterministic=deterministic,
            reset_memory=False,  # We handle reset above
        )

        # Final layer norm
        hidden_states = self.final_norm(hidden_states)

        # Output projection to vocabulary
        return self.output_projection(hidden_states)


class LanguageModelTrainer:
    """Trainer for language models using Titans."""

    def __init__(
        self,
        model: TitansLanguageModel,
        config: TitansConfig,
        tokenizer: TiktokenTokenizer,
        checkpoint_dir: Path | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Language model to train
            config: Training configuration
            tokenizer: Tokenizer instance
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer

        # Create optimizer
        self.optimizer = nnx.Optimizer(
            model,
            optax.adam(config.training.learning_rate),
            wrt=nnx.Param,
        )

        # Checkpoint manager
        if checkpoint_dir:
            self.checkpoint_manager = TitansCheckpoint(checkpoint_dir)
        else:
            self.checkpoint_manager = None

        # Training state
        self.step = 0
        self.epoch = 0

    def compute_loss(
        self,
        batch: dict[str, jax.Array],
        training: bool = True,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Compute language modeling loss.

        Args:
            batch: Batch with input_ids, target_ids, attention_mask
            training: Whether in training mode

        Returns:
            Tuple of (loss, metrics)
        """
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        attention_mask = batch["attention_mask"]

        # Forward pass
        logits: jax.Array = self.model(input_ids, deterministic=not training)

        # Compute cross-entropy loss
        log_probs: jax.Array = nnx.log_softmax(logits, axis=-1)

        # One-hot encode targets
        target_one_hot = nnx.one_hot(
            target_ids, num_classes=self.model.vocab_size, dtype=logits.dtype
        )

        # Compute loss per token
        token_losses: jax.Array = -jnp.sum(log_probs * target_one_hot, axis=-1)

        # Apply attention mask to ignore padding tokens
        masked_losses: jax.Array = token_losses * attention_mask

        # Average over non-padding tokens
        total_loss: jax.Array = jnp.sum(masked_losses)
        num_valid_tokens: jax.Array = jnp.sum(attention_mask)
        loss: jax.Array = total_loss / jnp.maximum(num_valid_tokens, 1)

        # Compute perplexity
        perplexity: jax.Array = jnp.exp(loss)

        # Compute accuracy (top-1)
        predictions: jax.Array = jnp.argmax(logits, axis=-1)
        correct_predictions: jax.Array = (predictions == target_ids) * attention_mask
        accuracy: jax.Array = jnp.sum(correct_predictions) / jnp.maximum(
            num_valid_tokens, 1
        )

        metrics = {
            "loss": loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "num_tokens": num_valid_tokens,
        }

        return loss, metrics

    def train_step(self, batch: dict[str, jax.Array]) -> dict[str, Any]:
        """Perform a single training step."""

        def loss_fn(model: TitansLanguageModel) -> jax.Array:
            # Use the model parameter, not self.model
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            attention_mask = batch["attention_mask"]

            # Forward pass with the model parameter - disable memory updates during gradient computation
            logits = model(
                input_ids, deterministic=True
            )  # Use deterministic=True to disable memory updates

            # Compute cross-entropy loss
            log_probs = nnx.log_softmax(logits, axis=-1)

            # One-hot encode targets
            target_one_hot = nnx.one_hot(
                target_ids, num_classes=model.vocab_size, dtype=logits.dtype
            )

            # Compute loss per token
            token_losses = -jnp.sum(log_probs * target_one_hot, axis=-1)

            # Apply attention mask to ignore padding tokens
            masked_losses = token_losses * attention_mask

            # Average over non-padding tokens
            total_loss = jnp.sum(masked_losses)
            num_valid_tokens = jnp.sum(attention_mask)
            return total_loss / jnp.maximum(num_valid_tokens, 1)

        # Compute gradients for main model parameters (memory updates disabled)
        _loss, grads = nnx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grads)

        # Now do a separate forward pass to trigger memory updates
        input_ids = batch["input_ids"]
        # Use deterministic=False to enable memory updates, but don't use gradients
        _ = self.model(input_ids, deterministic=False, reset_memory=False)

        self.step += 1

        # Compute metrics
        _, metrics = self.compute_loss(batch, training=False)

        return metrics

    def evaluate(self, eval_dataset: LanguageModelDataset) -> dict[str, float]:
        """Evaluate model on dataset."""
        total_loss = 0.0
        total_perplexity = 0.0
        total_accuracy = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in eval_dataset:
            _, metrics = self.compute_loss(batch, training=False)

            batch_tokens = metrics["num_tokens"]
            total_loss += metrics["loss"] * batch_tokens
            total_perplexity += metrics["perplexity"] * batch_tokens
            total_accuracy += metrics["accuracy"] * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

        if total_tokens > 0:
            return {
                "eval_loss": total_loss / total_tokens,
                "eval_perplexity": total_perplexity / total_tokens,
                "eval_accuracy": total_accuracy / total_tokens,
                "eval_num_batches": num_batches,
            }
        return {}

    def generate_text(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (None for no filtering)

        Returns:
            Generated text
        """
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        # Reset memory for generation
        self.model.titans_model.reset_memory()

        # Generate tokens
        generated_tokens = prompt_tokens.copy()

        for _ in range(max_length):
            # Current input
            input_ids = jnp.array([generated_tokens], dtype=jnp.int32)

            # Forward pass
            logits = self.model(input_ids, deterministic=True)

            # Get logits for next token (last position)
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
                # Set non-top-k logits to -inf
                next_token_logits = jnp.full_like(next_token_logits, -jnp.inf)
                next_token_logits = next_token_logits.at[top_k_indices].set(
                    top_k_logits
                )

            # Sample next token
            key = jax.random.key(len(generated_tokens))
            next_token = jax.random.categorical(key, next_token_logits)

            # Add to sequence
            generated_tokens.append(int(next_token))

            # Stop at EOS token
            if next_token == self.tokenizer.eos_token_id:
                break

        # Decode and return
        return self.tokenizer.decode(generated_tokens)


def create_sample_dataset() -> tuple[list[str], list[str]]:
    """Create sample text dataset for demonstration."""
    # Read Shakespeare text from file
    shakespeare_path = Path(__file__).parent / "data" / "shakespeare.txt"

    try:
        with open(shakespeare_path, encoding="utf-8") as f:
            shakespeare_text = f.read()
    except FileNotFoundError:
        print(f"Warning: Shakespeare file not found at {shakespeare_path}")
        print("Falling back to hardcoded sample data...")
        # Fallback to original hardcoded data
        train_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of artificial intelligence.",
            "Deep neural networks can learn complex patterns from data.",
            "Attention mechanisms have revolutionized natural language processing.",
            "Transformers use self-attention to model long-range dependencies.",
            "Language models can generate coherent and contextual text.",
            "The field of AI is rapidly evolving with new breakthroughs.",
            "Neural networks are inspired by the structure of the human brain.",
            "Gradient descent is a fundamental optimization algorithm.",
            "Large language models demonstrate emergent capabilities.",
        ] * 20  # Repeat for more training data

        val_texts = [
            "Artificial intelligence is changing how we interact with technology.",
            "Natural language understanding requires sophisticated models.",
            "The future of AI holds many exciting possibilities.",
        ] * 5

        return train_texts, val_texts

    # Clean and split the Shakespeare text
    # Split by double newlines to separate stanzas more naturally
    paragraphs = shakespeare_text.split("\n\n")

    # Filter out very short paragraphs and the header
    meaningful_texts: list[str] = []
    for i, para in enumerate(paragraphs):
        cleaned = para.strip()
        # Skip empty paragraphs
        if not cleaned:
            continue
        # Skip header (title and author)
        if i < 2:  # Skip first two sections which are title and author
            continue
        # Clean up the paragraph - remove extra whitespace
        cleaned = " ".join(cleaned.split())
        # Only keep paragraphs with at least 20 words
        if len(cleaned.split()) >= 20:
            meaningful_texts.append(cleaned)

    if len(meaningful_texts) == 0:
        print("No meaningful texts found, falling back to hardcoded data...")
        # Fallback if still no data
        train_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of artificial intelligence.",
            "Deep neural networks can learn complex patterns from data.",
            "Attention mechanisms have revolutionized natural language processing.",
            "Transformers use self-attention to model long-range dependencies.",
            "Language models can generate coherent and contextual text.",
            "The field of AI is rapidly evolving with new breakthroughs.",
            "Neural networks are inspired by the structure of the human brain.",
            "Gradient descent is a fundamental optimization algorithm.",
            "Large language models demonstrate emergent capabilities.",
        ] * 20  # Repeat for more training data

        val_texts = [
            "Artificial intelligence is changing how we interact with technology.",
            "Natural language understanding requires sophisticated models.",
            "The future of AI holds many exciting possibilities.",
        ] * 5

        return train_texts, val_texts

    # Split into train and validation sets (90% train, 10% validation)
    split_idx = int(0.9 * len(meaningful_texts))
    train_texts = meaningful_texts[:split_idx]
    val_texts = meaningful_texts[split_idx:]

    # Ensure we have enough validation data
    if len(val_texts) < 5:
        # Take last few training examples as validation
        val_texts = train_texts[-10:]
        train_texts = train_texts[:-10]

    return train_texts, val_texts


def language_modeling_example() -> None:
    """Complete example of language modeling with Titans."""
    print("Titans Language Modeling Example")
    print("=" * 40)

    # Configuration
    config = create_config_for_task("language_modeling", "small")
    config.training.learning_rate = 1e-3
    config.training.num_epochs = 10

    # Use proper GPT-2 vocabulary size for tiktoken
    vocab_size = 50257  # GPT-2 vocabulary size
    d_model = config.architecture.d_model

    print("Configuration:")
    print(f"  Variant: {config.variant}")
    print(f"  d_model: {d_model}")
    print(f"  vocab_size: {vocab_size}")

    # Create tokenizer
    tokenizer = TiktokenTokenizer(encoding_name="gpt2")

    # Create datasets
    print("\nCreating datasets...")
    train_texts, val_texts = create_sample_dataset()
    print(f"  Train texts: {len(train_texts)}")
    print(f"  Val texts: {len(val_texts)}")

    train_dataset = LanguageModelDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_length=128,
        batch_size=config.training.batch_size,
    )

    val_dataset = LanguageModelDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_length=128,
        batch_size=config.training.batch_size,
    )

    # Create model
    print("\nCreating model...")
    rngs = nnx.Rngs(config.seed)

    model = TitansLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        titans_config=config,
        rngs=rngs,
    )
    print(f"  Model created with {config.variant.upper()} variant")

    # Create trainer
    trainer = LanguageModelTrainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        checkpoint_dir=Path("language_model_checkpoints").absolute(),
    )

    # Training loop
    print("\nStarting training...\n")

    for epoch in range(config.training.num_epochs):
        print(f"Epoch {epoch + 1}/{config.training.num_epochs}")

        # Training
        epoch_metrics = []
        for batch_idx, batch in enumerate(train_dataset):
            metrics = trainer.train_step(batch)
            epoch_metrics.append(metrics)

            if batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}: Loss={metrics['loss']:.4f}, PPL={metrics['perplexity']:.2f}, Acc={metrics['accuracy']:.3f}"
                )

            if batch_idx >= 50:  # Limit for demo
                break

        # Evaluation
        print("  Evaluating...")
        eval_metrics = trainer.evaluate(val_dataset)
        if eval_metrics:
            print(
                f"  Eval: Loss={eval_metrics['eval_loss']:.4f}, PPL={eval_metrics['eval_perplexity']:.2f}, Acc={eval_metrics['eval_accuracy']:.3f}\n"
            )
        else:
            print("  Eval: No data processed\n")

    # Text generation
    print("Generating text...")
    prompts = [
        "The future of artificial intelligence",
        "Machine learning can",
        "Deep neural networks",
    ]

    for prompt in prompts:
        generated = trainer.generate_text(
            prompt=prompt,
            max_length=50,
            temperature=0.8,
            top_k=10,
        )
        print(f"  Prompt: '{prompt}'")
        print(f"  Generated: '{generated}'\n")

    # Save final model
    if trainer.checkpoint_manager:
        trainer.checkpoint_manager.save_model(
            trainer.model.titans_model,  # Save the Titans part
            trainer.step,
            config,
            metadata={"final_model": True},
        )
        print(f"Model saved to checkpoint at step {trainer.step}")

    print("Language modeling example completed!")


def memory_analysis_during_generation() -> None:
    """Analyze how memory evolves during text generation."""
    print("\nMemory Analysis During Generation")
    print("=" * 40)

    # Create simple model with SAME vocab size as main example
    config = create_config_for_task("language_modeling", "small")
    vocab_size = 50257  # Same as main example - GPT-2 vocab size

    model = TitansLanguageModel(
        vocab_size=vocab_size,
        d_model=config.architecture.d_model,
        titans_config=config,
        rngs=nnx.Rngs(42),
    )

    tokenizer = TiktokenTokenizer(encoding_name="gpt2")

    # Generate sequence and track memory
    prompt = "AI will change"
    print(f"Prompt: '{prompt}'")
    print("Tracking memory during generation:")

    input_tokens = tokenizer.encode(prompt)
    print(f"Encoded tokens: {input_tokens}")

    model.titans_model.reset_memory()
    memory_magnitudes: list[float] = []
    generated_tokens = input_tokens.copy()

    for step in range(10):
        # Forward pass
        input_ids = jnp.array([generated_tokens], dtype=jnp.int32)
        logits = model(input_ids, deterministic=False)

        # Sample next token
        next_token_logits = logits[0, -1, :]
        key = jax.random.key(step)
        next_token = jax.random.categorical(key, next_token_logits)
        generated_tokens.append(int(next_token))

        # Analyze memory - add debugging
        memory_state = model.titans_model.get_memory_state()
        if memory_state and hasattr(memory_state, "weights") and memory_state.weights:
            try:
                # Check for NaN in individual weights
                weight_magnitudes: list[float] = []
                for i, w in enumerate(memory_state.weights):
                    w_mag = float(jnp.sum(jnp.abs(w)))
                    weight_magnitudes.append(w_mag)
                    if jnp.isnan(w_mag):
                        print(f"    Warning: NaN in weight {i}")

                magnitude = sum(weight_magnitudes)
                memory_magnitudes.append(magnitude)
                print(
                    f"  Step {step + 1}: Token={next_token:3d}, Memory={magnitude:.4f}"
                )
            except Exception as e:
                print(f"  Step {step + 1}: Token={next_token:3d}, Memory=Error: {e}")
                memory_magnitudes.append(0.0)
        else:
            print(f"  Step {step + 1}: Token={next_token:3d}, Memory=No memory state")
            memory_magnitudes.append(0.0)

        if next_token == tokenizer.eos_token_id:
            break

    # Show memory evolution
    print("\nMemory evolution:")
    if len(memory_magnitudes) >= 2:
        valid_magnitudes = [m for m in memory_magnitudes if not jnp.isnan(m)]
        if len(valid_magnitudes) >= 2:
            initial_memory = valid_magnitudes[0]
            final_memory = valid_magnitudes[-1]
            growth_rate = (final_memory - initial_memory) / len(valid_magnitudes)
            print(f"  Initial: {initial_memory:.4f}")
            print(f"  Final: {final_memory:.4f}")
            print(f"  Growth rate: {growth_rate:.6f} per step")
        else:
            print("  No valid memory data to analyze")

    # Decode final sequence
    final_text = tokenizer.decode(generated_tokens)
    print(f"  Generated: '{final_text}'")


if __name__ == "__main__":
    # Run language modeling example
    language_modeling_example()

    # Run memory analysis
    memory_analysis_during_generation()
