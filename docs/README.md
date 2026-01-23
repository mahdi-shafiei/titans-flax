# Titans Documentation

This directory contains the documentation for the Titans project, built with MkDocs and Material theme.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install "titans-flax[docs]"
```

Or install the dependencies directly:

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-autorefs mike mkdocs-awesome-pages-plugin
```

### Development Server

Start the development server to preview documentation changes:

```bash
mkdocs serve
```

The documentation will be available at `http://localhost:8000` and will automatically reload when you make changes.

### Building Static Site

Build the static documentation site:

```bash
mkdocs build
```

This creates a `site/` directory with the static HTML files.

### Deploying to GitHub Pages

Deploy to GitHub Pages using Mike for versioning:

```bash
# Deploy latest version
mike deploy --push --update-aliases main latest

# Deploy a specific version
mike deploy --push --update-aliases v0.1.0 v0.1

# Set default version
mike set-default --push latest
```

## Documentation Structure

```
docs/
├── index.md                    # Homepage
├── getting-started/            # Installation and quick start
│   ├── installation.md
│   ├── quick-start.md
│   └── configuration.md
├── tutorials/                  # Step-by-step guides
│   ├── index.md
│   ├── understanding-titans.md
│   ├── basic-usage.md
│   ├── memory-mechanisms.md
│   ├── architecture-variants.md
│   ├── training-models.md
│   ├── test-time-adaptation.md
│   └── performance-optimization.md
├── examples/                   # Practical examples
│   ├── index.md
│   ├── language-modeling.md
│   ├── long-context.md
│   ├── memory-analysis.md
│   ├── benchmarking.md
│   └── research-reproduction.md
├── api/                        # API reference (auto-generated)
│   ├── index.md
│   ├── core.md
│   ├── neural-memory.md
│   ├── attention-modules.md
│   ├── variants.md
│   ├── config.md
│   ├── training.md
│   ├── optimization-utilities.md
│   ├── research-utilities.md
│   ├── benchmark.md
│   └── serialization.md
├── javascripts/               # Custom JavaScript
│   └── mathjax.js
├── stylesheets/              # Custom CSS
│   └── extra.css
└── README.md                 # This file
```

## Writing Documentation

### Markdown Files

All documentation is written in Markdown with some extensions:

- **Admonitions**: `!!! tip "Title"`
- **Code highlighting**: Fenced code blocks with language specification
- **Math**: LaTeX math with `\(inline\)` and `\[display\]`
- **Tabs**: Content tabs using `=== "Tab Name"`

### API Documentation

API documentation is automatically generated from docstrings using mkdocstrings:

```markdown
::: titans.module_name
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
```

### Mathematical Notation

Use LaTeX notation for math:

```markdown
Inline math: \(f(x) = x^2\)

Display math:
\[
\memory_{t+1} = \memory_t - \alpha \grad_{\memory_t} \loss(\memory_t, \vec{x}_t)
\]
```

Custom macros are defined in `javascripts/mathjax.js`:
- `\memory` for memory notation
- `\attn` for attention
- `\grad` for gradients
- And many more...

### Code Examples

Include runnable code examples:

```python
from titans import get_config, create_titans_model
import jax
import jax.numpy as jnp

config = get_config("medium")  # small, medium, large, long_context, efficient
model = create_titans_model(
    variant=config.variant,
    d_model=config.architecture.d_model,
    seed=42
)
x = jax.random.normal(jax.random.key(42), (1, 128, 512))
output = model(x, deterministic=False)

# Task-specific configuration
from titans.core import create_titans_model
```

### Cross-References

Link to other pages and sections:

```markdown
See the [Installation Guide](getting-started/installation.md)
Check out [API Reference](api/core.md#create_titans_model)
```

## Contributing

### Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add the page to `mkdocs.yml` navigation
3. Update any relevant index pages
4. Test with `mkdocs serve`

### Updating API Documentation

API documentation is automatically generated from source code docstrings. To update:

1. Modify docstrings in the source code
2. Rebuild documentation with `mkdocs build`
3. The changes will appear automatically

### Style Guidelines

- Use clear, concise language
- Include code examples for all concepts
- Add mathematical notation where helpful
- Use admonitions for tips, warnings, and notes
- Test all code examples before publishing

### Review Process

Before publishing documentation changes:

1. Test locally with `mkdocs serve`
2. Check all links and references
3. Verify code examples run correctly
4. Review for clarity and completeness
5. Build with `mkdocs build` to check for errors

## Troubleshooting

### Common Issues

**MkDocs build fails**: Check `mkdocs.yml` syntax and ensure all referenced files exist.

**API documentation not updating**: Restart the development server after changing source code.

**Math not rendering**: Check MathJax configuration in `javascripts/mathjax.js`.

**Styling issues**: Verify CSS in `stylesheets/extra.css` and Material theme compatibility.

### Getting Help

- Check the [MkDocs documentation](https://www.mkdocs.org/)
- Review [Material theme docs](https://squidfunk.github.io/mkdocs-material/)
- See [mkdocstrings documentation](https://mkdocstrings.github.io/)

## Automation

### GitHub Actions

The project includes GitHub Actions to automatically:

- Build documentation on pull requests
- Deploy to GitHub Pages on main branch updates
- Check for broken links and missing references

### Pre-commit Hooks

Documentation linting is included in pre-commit hooks:

```bash
pre-commit install
```

This will check:
- Markdown formatting
- Link validity
- Spelling (if configured)
- Code example syntax
