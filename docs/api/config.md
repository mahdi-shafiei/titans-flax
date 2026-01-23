# Configuration

Configuration management for Titans models including predefined configurations and task-specific setups.

::: titans.config
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Key Classes

- **TitansConfig** - Complete configuration class with validation
- **ArchitectureConfig** - Architecture-specific settings
- **MemoryConfig** - Memory module configuration
- **AttentionConfig** - Attention mechanism settings
- **TrainingConfig** - Training hyperparameters

## Functions

- **get_config()** - Get predefined configurations (small, medium, large)
- **create_config_for_task()** - Task-specific configurations (language_modeling, etc.)
