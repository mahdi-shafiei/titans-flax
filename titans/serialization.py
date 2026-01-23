"""Serialization utilities for Titans models.

This module provides complete serialization and checkpointing
capabilities for Titans models, including memory state management
and model persistence.
"""

import json
from pathlib import Path
import shutil
import time
from typing import Any

from flax import nnx
import jax.numpy as jnp
import orbax.checkpoint as ocp

from titans.config import TitansConfig
from titans.core import TitansModel
from titans.neural_memory import MemoryState


class TitansCheckpoint:
    """Secure checkpoint manager for Titans models using Orbax."""

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        # Orbax requires absolute paths
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Orbax checkpoint manager
        self.orbax_manager = ocp.CheckpointManager(
            self.checkpoint_dir / "orbax_checkpoints",
            ocp.PyTreeCheckpointer(),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=5,
                create=True,
            ),
        )

    def save_model(
        self,
        model: TitansModel,
        step: int,
        config: TitansConfig,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        optimizer: nnx.Optimizer | None = None,
    ) -> Path:
        """Save a complete model checkpoint using Orbax.

        Both model AND optimizer state should be saved for proper training resumption.

        Args:
            model: Titans model to save
            step: Training step number
            config: Model configuration
            metrics: Optional training metrics
            metadata: Optional additional metadata
            optimizer: Optional optimizer to save state for training resumption

        Returns:
            Path to the saved checkpoint
        """
        # Runtime validation for defensive programming
        if not isinstance(model, TitansModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"Expected TitansModel, got {type(model)}"
            raise TypeError(msg)
        if step < 0:
            msg = f"step must be non-negative, got {step}"
            raise ValueError(msg)

        checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:06d}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model parameters using Orbax
        model_state = nnx.state(model)
        self.orbax_manager.save(step, model_state)

        # Save optimizer state if provided (CRITICAL for training resumption)
        if optimizer is not None:
            optimizer_checkpointer = ocp.PyTreeCheckpointer()
            optimizer_state = nnx.state(optimizer)
            optimizer_state_path = checkpoint_path / "optimizer_state"
            # Remove existing optimizer state if it exists to allow re-saving
            if optimizer_state_path.exists():
                shutil.rmtree(optimizer_state_path)
            optimizer_checkpointer.save(optimizer_state_path, optimizer_state)

        # Save memory state using Orbax (if it exists)
        memory_state = model.get_memory_state()
        if memory_state:
            memory_checkpointer = ocp.PyTreeCheckpointer()
            memory_data = {
                "weights": memory_state.weights,
                "momentum": memory_state.momentum,
            }
            memory_checkpointer.save(checkpoint_path / "memory_state", memory_data)

        # Save configuration
        config_path = checkpoint_path / "config.json"
        config.save(config_path)

        # Save metadata
        checkpoint_metadata = {
            "step": step,
            "timestamp": time.time(),
            "variant": model.variant,
            "d_model": model.d_model,
            "param_count": model.count_params(),
            "metrics": metrics or {},
            "metadata": metadata or {},
            "has_optimizer_state": optimizer is not None,
        }

        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(checkpoint_metadata, f, indent=2)

        return checkpoint_path

    def load_model(
        self,
        step: int | None = None,
        checkpoint_path: Path | None = None,
        optimizer: nnx.Optimizer | None = None,
    ) -> tuple[TitansModel, TitansConfig, dict[str, Any]]:
        """Load a model checkpoint.

        Restores both model AND optimizer state for proper training resumption.

        Args:
            step: Training step to load (loads latest if None)
            checkpoint_path: Specific checkpoint path to load from
            optimizer: Optional optimizer to restore state into

        Returns:
            Tuple of (model, config, metadata)
        """
        if checkpoint_path is None:
            if step is None:
                step = self.orbax_manager.latest_step()
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{step:06d}"

        if not checkpoint_path.exists():
            msg = f"Checkpoint not found: {checkpoint_path}"
            raise FileNotFoundError(msg)

        # Load configuration
        config_path = checkpoint_path / "config.json"
        if not config_path.exists():
            msg = f"Config not found: {config_path}"
            raise FileNotFoundError(msg)
        config = TitansConfig.load(config_path)

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        if not metadata_path.exists():
            msg = f"Metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Create model with same configuration
        model = TitansModel(
            variant=config.variant,
            d_model=config.architecture.d_model,
            num_heads=config.attention.num_heads,
            memory_depth=config.memory.memory_depth,
            num_persistent_tokens=config.memory.num_persistent_tokens,
            chunk_size=config.architecture.chunk_size,
            window_size=config.attention.window_size or 256,
            dropout_rate=config.attention.dropout_rate,
            rngs=nnx.Rngs(config.seed),
            dtype=config.architecture.get_dtype(),
        )

        # Load model parameters
        if step is None:
            step = metadata["step"]

        restored_state = self.orbax_manager.restore(step)
        nnx.update(model, restored_state)

        # Load optimizer state if it exists and optimizer is provided
        # (CRITICAL for training resumption)
        optimizer_state_path = checkpoint_path / "optimizer_state"
        if optimizer is not None and optimizer_state_path.exists():
            optimizer_checkpointer = ocp.PyTreeCheckpointer()
            restored_opt_state = optimizer_checkpointer.restore(optimizer_state_path)
            nnx.update(optimizer, restored_opt_state)

        # Load memory state if it exists
        memory_state_path = checkpoint_path / "memory_state"
        if memory_state_path.exists():
            memory_checkpointer = ocp.PyTreeCheckpointer()
            memory_data = memory_checkpointer.restore(memory_state_path)
            memory_state = MemoryState(
                weights=memory_data["weights"],
                momentum=memory_data["momentum"],
            )
            model.set_memory_state(memory_state)

        return model, config, metadata

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all available checkpoints.

        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []

        for checkpoint_dir in self.checkpoint_dir.glob("checkpoint_*"):
            if checkpoint_dir.is_dir():
                metadata_path = checkpoint_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        metadata["path"] = str(checkpoint_dir)
                        checkpoints.append(metadata)
                    except (json.JSONDecodeError, OSError):
                        continue

        checkpoints.sort(key=lambda x: x.get("timestamp", 0))
        return checkpoints

    def cleanup_old_checkpoints(self, keep_last: int = 5) -> None:
        """Clean up old checkpoints, keeping only the most recent ones.

        Args:
            keep_last: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_last:
            return

        # Sort by timestamp and remove oldest
        checkpoints.sort(key=lambda x: x.get("timestamp", 0))
        to_remove = checkpoints[:-keep_last]

        for checkpoint in to_remove:
            checkpoint_path = Path(checkpoint["path"])
            if checkpoint_path.exists():
                # Remove checkpoint directory
                shutil.rmtree(checkpoint_path)


class ModelExporter:
    """Export Titans models for inference optimization."""

    @staticmethod
    def export_for_inference(
        model: TitansModel,
        output_path: Path,
        sample_input_shape: tuple[int, ...] = (1, 512, 128),
        include_memory_state: bool = True,
    ) -> None:
        """Export model for optimized inference.

        Args:
            model: Model to export
            output_path: Path to save exported model
            sample_input_shape: Shape for compilation optimization
            include_memory_state: Whether to include memory state
        """
        # Runtime validation for defensive programming
        if not isinstance(model, TitansModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"Expected TitansModel, got {type(model)}"
            raise TypeError(msg)

        # Orbax requires absolute paths
        output_path = Path(output_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        model_state = nnx.state(model)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(output_path / "model_state", model_state)

        # Save memory state if requested
        if include_memory_state:
            memory_state = model.get_memory_state()
            if memory_state:
                memory_data = {
                    "weights": memory_state.weights,
                    "momentum": memory_state.momentum,
                }
                checkpointer.save(output_path / "memory_state", memory_data)

        # Save model metadata
        export_metadata = {
            "variant": model.variant,
            "d_model": model.d_model,
            "param_count": model.count_params(),
            "sample_input_shape": sample_input_shape,
            "include_memory_state": include_memory_state,
            "export_timestamp": time.time(),
        }

        with open(output_path / "export_metadata.json", "w") as f:
            json.dump(export_metadata, f, indent=2)

    @staticmethod
    def load_for_inference(
        export_path: Path,
        rngs: nnx.Rngs,
    ) -> TitansModel:
        """Load exported model for inference.

        Args:
            export_path: Path to exported model
            rngs: Random number generators

        Returns:
            Loaded model ready for inference
        """
        # Orbax requires absolute paths
        export_path = Path(export_path).resolve()

        if not export_path.exists():
            msg = f"Export path not found: {export_path}"
            raise FileNotFoundError(msg)

        # Load metadata
        metadata_path = export_path / "export_metadata.json"
        if not metadata_path.exists():
            msg = f"Export metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)

        with open(metadata_path) as f:
            metadata = json.load(f)

        # Create model with same architecture
        # Note: We need to infer the constructor parameters from metadata
        # This is a simplified approach - in practice, you might want to save
        # the full config alongside the export
        model = TitansModel(
            variant=metadata["variant"],
            d_model=metadata["d_model"],
            rngs=rngs,
        )

        # Load model state
        checkpointer = ocp.PyTreeCheckpointer()
        model_state = checkpointer.restore(export_path / "model_state")
        nnx.update(model, model_state)

        # Load memory state if available
        memory_state_path = export_path / "memory_state"
        if memory_state_path.exists():
            memory_data = checkpointer.restore(memory_state_path)
            memory_state = MemoryState(
                weights=memory_data["weights"],
                momentum=memory_data["momentum"],
            )
            model.set_memory_state(memory_state)

        return model


class MemoryStateManager:
    """Manage persistent memory states for Titans models."""

    def __init__(self, storage_dir: Path):
        """Initialize memory state manager.

        Args:
            storage_dir: Directory to store memory states
        """
        # Orbax requires absolute paths
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_memory_state(
        self,
        memory_state: MemoryState,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a memory state with metadata.

        Args:
            memory_state: Memory state to save
            name: Name for the memory state
            metadata: Optional metadata

        Returns:
            Path to saved memory state
        """
        # Runtime validation for defensive programming
        if not isinstance(memory_state, MemoryState):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"Expected MemoryState, got {type(memory_state)}"
            raise TypeError(msg)
        if not name:
            msg = "name cannot be empty"
            raise ValueError(msg)

        state_path = self.storage_dir / f"{name}"
        metadata_path = self.storage_dir / f"{name}_metadata.json"

        # Save memory state using Orbax
        memory_data = {
            "weights": memory_state.weights,
            "momentum": memory_state.momentum,
        }

        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(state_path, memory_data)

        # Save metadata
        full_metadata = {
            "name": name,
            "timestamp": time.time(),
            "num_layers": len(memory_state.weights),
            "weight_shapes": [w.shape for w in memory_state.weights],
            "metadata": metadata or {},
        }

        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        return state_path

    def load_memory_state(self, name: str) -> tuple[MemoryState, dict[str, Any]]:
        """Load a memory state with its metadata.

        Args:
            name: Name of the memory state to load

        Returns:
            Tuple of (memory_state, metadata)
        """
        if not name:
            msg = "name cannot be empty"
            raise ValueError(msg)

        state_path = self.storage_dir / f"{name}"
        metadata_path = self.storage_dir / f"{name}_metadata.json"

        if not state_path.exists():
            msg = f"Memory state not found: {state_path}"
            raise FileNotFoundError(msg)
        if not metadata_path.exists():
            msg = f"Memory metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)

        # Load memory state using Orbax
        checkpointer = ocp.PyTreeCheckpointer()
        memory_data = checkpointer.restore(state_path)

        memory_state = MemoryState(
            weights=memory_data["weights"],
            momentum=memory_data["momentum"],
        )

        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)

        return memory_state, metadata

    def list_memory_states(self) -> list[dict[str, Any]]:
        """List all available memory states.

        Returns:
            List of memory state metadata
        """
        states = []

        for metadata_path in self.storage_dir.glob("*_metadata.json"):
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                states.append(metadata)
            except (json.JSONDecodeError, OSError):
                continue

        states.sort(key=lambda x: x.get("timestamp", 0))
        return states

    def compare_memory_states(
        self,
        name1: str,
        name2: str,
    ) -> dict[str, float]:
        """Compare two memory states.

        Args:
            name1: First memory state name
            name2: Second memory state name

        Returns:
            Dictionary of comparison metrics
        """
        state1, _ = self.load_memory_state(name1)
        state2, _ = self.load_memory_state(name2)

        if len(state1.weights) != len(state2.weights):
            msg = "Memory states have different numbers of layers"
            raise ValueError(msg)

        # Calculate differences
        weight_diffs = []
        momentum_diffs = []

        for w1, w2 in zip(state1.weights, state2.weights):
            if w1.shape != w2.shape:
                msg = f"Weight shapes don't match: {w1.shape} vs {w2.shape}"
                raise ValueError(msg)
            diff = jnp.mean(jnp.abs(w1 - w2))
            weight_diffs.append(float(diff))

        for m1, m2 in zip(state1.momentum, state2.momentum):
            if m1.shape != m2.shape:
                msg = f"Momentum shapes don't match: {m1.shape} vs {m2.shape}"
                raise ValueError(msg)
            diff = jnp.mean(jnp.abs(m1 - m2))
            momentum_diffs.append(float(diff))

        return {
            "avg_weight_diff": float(jnp.mean(jnp.array(weight_diffs))),
            "max_weight_diff": float(jnp.max(jnp.array(weight_diffs))),
            "avg_momentum_diff": float(jnp.mean(jnp.array(momentum_diffs))),
            "max_momentum_diff": float(jnp.max(jnp.array(momentum_diffs))),
            "total_param_diff": float(
                jnp.sum(jnp.array(weight_diffs + momentum_diffs))
            ),
        }
