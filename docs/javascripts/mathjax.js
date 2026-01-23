window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    tags: 'ams',
    packages: {'[+]': ['ams', 'newcommand', 'color']},
    macros: {
      // Common mathematical notation for Titans
      Real: "{\\mathbb{R}}",
      Complex: "{\\mathbb{C}}",
      Natural: "{\\mathbb{N}}",
      Integer: "{\\mathbb{Z}}",

      // Vectors and matrices
      vec: ["\\mathbf{#1}", 1],
      mat: ["\\mathbf{#1}", 1],

      // Common functions
      softmax: "{\\text{softmax}}",
      relu: "{\\text{ReLU}}",
      gelu: "{\\text{GELU}}",

      // Attention notation
      attn: "{\\text{Attention}}",
      query: ["\\mathbf{Q}", 0],
      key: ["\\mathbf{K}", 0],
      value: ["\\mathbf{V}", 0],

      // Memory notation
      memory: ["\\mathcal{M}", 0],
      memupdate: ["\\Delta \\mathcal{M}", 0],

      // Commonly used symbols
      loss: ["\\mathcal{L}", 0],
      params: ["\\theta", 0],
      grad: ["\\nabla", 0],

      // Sequence notation
      seq: ["\\mathbf{x}_{1:T}", 0],
      hidden: ["\\mathbf{h}", 0],

      // Probability
      prob: ["\\mathbb{P}", 0],
      expect: ["\\mathbb{E}", 0],

      // Common operations
      concat: ["{\\text{concat}(#1)}", 1],
      norm: ["{\\|#1\\|}", 1],

      // Architecture specific
      mac: ["{\\text{MAC}}", 0],
      mag: ["{\\text{MAG}}", 0],
      mal: ["{\\text{MAL}}", 0],
      lmm: ["{\\text{LMM}}", 0]
    }
  },
  options: {
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_process",
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  svg: {
    fontCache: 'global'
  },
  loader: {
    load: ['[tex]/ams', '[tex]/newcommand', '[tex]/color']
  }
};

document$.subscribe(() => {
  MathJax.typesetPromise()
});
