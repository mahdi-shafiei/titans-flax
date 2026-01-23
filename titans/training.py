"""Training utilities and loops for Titans models.

This module provides complete training infrastructure for Titans models,
including data management, loss computation, and evaluation metrics.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False
    wandb = None

from .config import TitansConfig, get_config
from .core import MEMORY_REGULARIZATION_WEIGHT, TitansModel, create_titans_model
from .serialization import MemoryStateManager, TitansCheckpoint


@dataclass
class TrainingMetrics:
    """Container for training metrics and state."""

    # Training configuration
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0

    # Training metrics
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    train_accuracy_history: list[float] = field(default_factory=list)
    val_accuracy_history: list[float] = field(default_factory=list)

    # Learning rate history
    lr_history: list[float] = field(default_factory=list)

    # Timing information
    step_times: list[float] = field(default_factory=list)

    # Memory evolution tracking
    memory_norm_history: list[float] = field(default_factory=list)

    def get_summary(self) -> dict[str, Any]:
        """Get training summary."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "avg_step_time": (np.mean(self.step_times) if self.step_times else 0.0),
            "current_train_loss": (
                self.train_loss_history[-1] if self.train_loss_history else None
            ),
            "current_val_loss": (
                self.val_loss_history[-1] if self.val_loss_history else None
            ),
            "current_memory_norm": (
                self.memory_norm_history[-1] if self.memory_norm_history else None
            ),
        }


class LearningRateScheduler:
    """Learning rate scheduling utilities."""

    @staticmethod
    def cosine_decay_with_warmup(
        init_lr: float,
        max_lr: float,
        warmup_steps: int,
        decay_steps: int,
        min_lr: float = 0.0,
    ) -> Callable[[Any], float]:
        """Cosine decay with linear warmup."""

        def schedule(step: Any) -> float:
            step = float(step)
            if step < warmup_steps:
                # Linear warmup
                return float(init_lr + (max_lr - init_lr) * step / warmup_steps)
            # Cosine decay
            progress = (step - warmup_steps) / (decay_steps - warmup_steps)
            progress = jnp.minimum(progress, 1.0)
            return float(
                min_lr + (max_lr - min_lr) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
            )

        return schedule

    @staticmethod
    def exponential_decay(
        init_lr: float,
        decay_rate: float,
        decay_steps: int,
        staircase: bool = False,
    ) -> Callable[[Any], float]:
        """Exponential decay schedule."""

        def schedule(step: Any) -> float:
            step = float(step)
            if staircase:
                step = jnp.floor(step / decay_steps) * decay_steps
            return float(init_lr * (decay_rate ** (step / decay_steps)))

        return schedule


class TitansTrainer:
    """Advanced trainer for Titans models with proper memory integration."""

    def __init__(
        self,
        model: TitansModel,
        config: TitansConfig,
        train_dataset: Iterator[dict[str, jax.Array]],
        val_dataset: Iterator[dict[str, jax.Array]] | None = None,
        checkpoint_dir: Path | None = None,
        use_wandb: bool = False,
        wandb_project: str = "titans",
    ):
        """Initialize trainer."""
        # Runtime validation for defensive programming
        if not isinstance(model, TitansModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"Expected TitansModel, got {type(model)}"
            raise TypeError(msg)

        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initialize training state
        self.training_state = TrainingMetrics()

        # Setup optimizer with learning rate schedule
        self.lr_schedule = self._create_lr_schedule()
        self.optimizer = self._create_optimizer()

        # Setup checkpointing
        if checkpoint_dir:
            self.checkpoint_manager = TitansCheckpoint(checkpoint_dir)
            self.memory_manager = MemoryStateManager(checkpoint_dir / "memory_states")
        else:
            self.checkpoint_manager = None
            self.memory_manager = None

        # Setup logging
        self.use_wandb = use_wandb and wandb_available
        if self.use_wandb and wandb is not None:
            try:
                wandb.init(
                    project=wandb_project,
                    config={
                        "d_model": model.d_model,
                        "variant": model.variant,
                        "learning_rate": config.training.learning_rate,
                        "optimizer": "adam_with_schedule",
                        "num_epochs": config.training.num_epochs,
                    },
                )
            except Exception:
                # Wandb not configured (no API key), disable it
                self.use_wandb = False

    def _create_lr_schedule(self) -> Callable[[Any], float]:
        """Create learning rate schedule."""
        return LearningRateScheduler.cosine_decay_with_warmup(
            init_lr=0.0,
            max_lr=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            decay_steps=10000,
            min_lr=self.config.training.learning_rate * 0.01,
        )

    def _create_optimizer(self) -> nnx.Optimizer:
        """Create optimizer with learning rate schedule."""
        tx_components = []

        # Gradient clipping
        if self.config.training.gradient_clip_norm:
            tx_components.append(
                optax.clip_by_global_norm(self.config.training.gradient_clip_norm)
            )

        # Adam optimizer with learning rate schedule
        tx_components.append(optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8))

        # Learning rate scaling
        tx_components.append(optax.scale_by_schedule(self.lr_schedule))

        # Final scaling (negative for gradient descent)
        tx_components.append(optax.scale(-1.0))

        tx = optax.chain(*tx_components)

        # UPDATED for Flax 0.11.0+: wrt parameter is now required
        return nnx.Optimizer(self.model, tx, wrt=nnx.Param)

    def compute_loss(
        self,
        batch: dict[str, jax.Array],
        training: bool = True,
    ) -> tuple[float, dict[str, Any]]:
        """Compute loss for a batch with proper memory integration.

        Args:
            batch: Batch of data
            training: Whether in training mode

        Returns:
            Tuple of (loss, metrics)
        """
        inputs = batch["inputs"]
        targets = batch["targets"]

        # Validate shapes
        if inputs.shape[:-1] != targets.shape[:-1]:
            msg = f"Input/target shape mismatch: {inputs.shape} vs {targets.shape}"
            raise ValueError(msg)

        # FIXED: Forward pass with integrated memory updates
        # Memory updates are now part of the computation graph
        predictions = self.model(inputs, deterministic=not training)

        # Compute loss (MSE for this example)
        loss = jnp.mean((predictions - targets) ** 2)

        # Compute additional metrics
        mae = jnp.mean(jnp.abs(predictions - targets))

        # Memory analysis
        memory_norm = self.model.get_memory_norm()

        metrics = {
            "loss": float(loss),
            "mae": float(mae),
            "memory_norm": memory_norm,
            "predictions_mean": float(jnp.mean(predictions)),
            "predictions_std": float(jnp.std(predictions)),
        }

        return float(loss), metrics

    def train_step(self, batch: dict[str, jax.Array]) -> dict[str, Any]:
        """Perform a single training step with proper memory integration."""
        start_time = time.time()

        # FIXED: Integrated loss function that includes memory updates in gradient computation
        def loss_fn(model: TitansModel) -> jax.Array:
            inputs = batch["inputs"]
            targets = batch["targets"]

            # Forward pass with memory updates as part of computation graph
            predictions = model(inputs, deterministic=False)

            # Compute loss
            loss = jnp.mean((predictions - targets) ** 2)

            # Optional: Add memory regularization
            memory_norm = model.get_memory_norm()
            memory_reg = MEMORY_REGULARIZATION_WEIGHT * memory_norm

            return loss + memory_reg

        # Compute gradients - memory updates are now included
        # UPDATED for Flax 0.11.0+: use self.model instead of self.optimizer.model
        _loss, grads = nnx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grads)  # Pass both model and grads

        # Update training state
        self.training_state.step += 1
        step_time = time.time() - start_time
        self.training_state.step_times.append(step_time)

        # Get current learning rate
        current_lr = self.lr_schedule(self.training_state.step)
        self.training_state.lr_history.append(current_lr)

        # Compute metrics (separate evaluation call)
        _, metrics = self.compute_loss(batch, training=False)
        metrics["learning_rate"] = current_lr
        metrics["step_time"] = step_time

        # Update loss and memory history
        self.training_state.train_loss_history.append(metrics["loss"])
        self.training_state.memory_norm_history.append(metrics["memory_norm"])

        return metrics

    def evaluate(self, num_batches: int | None = None) -> dict[str, Any]:
        """Evaluate model on validation set."""
        if self.val_dataset is None:
            return {}

        eval_metrics: list[dict[str, Any]] = []

        for batch_idx, batch in enumerate(self.val_dataset):
            if num_batches and batch_idx >= num_batches:
                break

            _, metrics = self.compute_loss(batch, training=False)
            eval_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        if eval_metrics:
            for key in eval_metrics[0]:
                avg_metrics[f"val_{key}"] = float(
                    jnp.mean(jnp.array([m[key] for m in eval_metrics]))
                )

            # Update validation history
            self.training_state.val_loss_history.append(avg_metrics["val_loss"])

            # Update best metrics
            if avg_metrics["val_loss"] < self.training_state.best_val_loss:
                self.training_state.best_val_loss = avg_metrics["val_loss"]
                self.training_state.epochs_without_improvement = 0
            else:
                self.training_state.epochs_without_improvement += 1

        return avg_metrics

    def train_epoch(self) -> dict[str, Any]:
        """Train for one epoch."""
        epoch_metrics = []

        # Training loop
        for batch_idx, batch in enumerate(self.train_dataset):
            metrics = self.train_step(batch)
            epoch_metrics.append(metrics)

            # Logging
            if self.training_state.step % self.config.training.log_interval == 0:
                avg_metrics = {
                    key: float(
                        jnp.mean(jnp.array([m[key] for m in epoch_metrics[-10:]]))
                    )
                    for key in metrics
                }

                if self.use_wandb and wandb is not None:
                    wandb.log(avg_metrics, step=self.training_state.step)

            # Evaluation
            if (
                self.training_state.step % self.config.training.eval_interval == 0
                and self.val_dataset is not None
            ):
                val_metrics = self.evaluate(num_batches=10)

                if val_metrics:
                    pass

                if self.use_wandb and wandb is not None:
                    wandb.log(val_metrics, step=self.training_state.step)

            # Checkpointing: save both model AND optimizer state for proper training resumption
            if (
                self.checkpoint_manager
                and self.training_state.step % self.config.training.save_interval == 0
            ):
                recent_metrics = {
                    key: float(
                        jnp.mean(jnp.array([m[key] for m in epoch_metrics[-10:]]))
                    )
                    for key in metrics
                }

                self.checkpoint_manager.save_model(
                    self.model,
                    self.training_state.step,
                    self.config,
                    metrics=recent_metrics,
                    optimizer=self.optimizer,  # Save optimizer state
                )

                # Save memory state separately
                if self.memory_manager:
                    memory_state = self.model.get_memory_state()
                    if memory_state:
                        self.memory_manager.save_memory_state(
                            memory_state,
                            f"step_{self.training_state.step}",
                            metadata={
                                "step": self.training_state.step,
                                "metrics": recent_metrics,
                            },
                        )

            # Break if we've seen enough batches for this epoch
            if batch_idx >= 1000:  # Adjust based on dataset size
                break

        # Epoch summary
        self.training_state.epoch += 1

        # Return epoch summary
        if epoch_metrics:
            return {
                key: float(jnp.mean(jnp.array([m[key] for m in epoch_metrics])))
                for key in epoch_metrics[0]
            }
        return {"loss": 0.0, "mae": 0.0, "memory_norm": 0.0}

    def train(self, num_epochs: int | None = None) -> dict[str, Any]:
        """Train the model."""
        if num_epochs is None:
            num_epochs = self.config.training.num_epochs

        training_start_time = time.time()

        try:
            for epoch in range(num_epochs):
                # Reset memory at start of each epoch for consistency
                if epoch > 0:  # Keep memory from first epoch
                    self.model.reset_memory()

                epoch_metrics = self.train_epoch()

                if self.use_wandb and wandb is not None:
                    wandb.log(
                        {
                            **epoch_metrics,
                            "epoch": self.training_state.epoch,
                        }
                    )

                # Early stopping check
                if (
                    self.training_state.epochs_without_improvement >= 5 and epoch > 2
                ):  # Allow some warmup
                    break

        except KeyboardInterrupt:
            pass

        finally:
            training_time = time.time() - training_start_time

            # Final evaluation
            if self.val_dataset:
                final_val_metrics = self.evaluate()
                if final_val_metrics:
                    pass

            # Save final checkpoint (including optimizer state for resumption)
            if self.checkpoint_manager:
                self.checkpoint_manager.save_model(
                    self.model,
                    self.training_state.step,
                    self.config,
                    metadata={"training_time": training_time, "final": True},
                    optimizer=self.optimizer,  # Save optimizer state
                )

            # Training summary
            summary = {
                **self.training_state.get_summary(),
                "total_training_time": training_time,
                "epochs_completed": self.training_state.epoch,
            }

            if self.use_wandb and wandb is not None:
                wandb.log(summary)
                wandb.finish()

        return summary

    def resume_from_checkpoint(self, step: int | None = None) -> None:
        """Resume training from a checkpoint.

        Restores both model AND optimizer state for smooth training resumption.
        """
        if not self.checkpoint_manager:
            msg = "No checkpoint manager configured"
            raise ValueError(msg)

        # Load model and restore optimizer state
        # Passing the optimizer ensures its state is also restored
        loaded_model, _, metadata = self.checkpoint_manager.load_model(
            step, optimizer=self.optimizer
        )

        # Update model state
        nnx.update(self.model, nnx.state(loaded_model))

        # Update training state
        self.training_state.step = metadata["step"]
        if "metrics" in metadata:
            metrics = metadata["metrics"]
            if "loss" in metrics:
                self.training_state.train_loss_history.append(metrics["loss"])


def create_synthetic_dataset(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_batches: int,
) -> Iterator[dict[str, jax.Array]]:
    """Create a synthetic dataset for testing."""
    for i in range(num_batches):
        key = jax.random.key(i)

        # Generate input sequence with some structure
        inputs = jax.random.normal(key, (batch_size, seq_len, d_model))

        # Generate targets with relationship to inputs (simple transformation)
        noise_key = jax.random.split(key)[1]
        targets = inputs + 0.1 * jax.random.normal(
            noise_key, (batch_size, seq_len, d_model)
        )

        yield {
            "inputs": inputs,
            "targets": targets,
        }


def training_example() -> None:
    """Example of training a Titans model with fixed architecture."""
    # Configuration
    config = get_config("small")
    config.training.num_epochs = 3
    config.training.log_interval = 10
    config.training.eval_interval = 50
    config.training.save_interval = 100

    # Create model
    model = create_titans_model(
        variant=config.variant,
        d_model=config.architecture.d_model,
        memory_depth=config.memory.memory_depth,
        seed=config.seed,
    )

    # Create datasets
    train_dataset = create_synthetic_dataset(
        batch_size=config.training.batch_size,
        seq_len=64,
        d_model=config.architecture.d_model,
        num_batches=200,
    )

    val_dataset = create_synthetic_dataset(
        batch_size=config.training.batch_size,
        seq_len=64,
        d_model=config.architecture.d_model,
        num_batches=20,
    )

    # Create trainer
    trainer = TitansTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=Path("training_example").resolve(),
        use_wandb=False,
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    training_example()
