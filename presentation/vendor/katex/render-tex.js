(function () {
  function renderAll() {
    if (!window.katex) return;
    document.querySelectorAll('.tex').forEach(function (node) {
      if (node.dataset.rendered === '1') return;
      var source = node.dataset.tex || node.textContent;
      var displayMode = node.classList.contains('display');
      window.katex.render(source, node, {
        displayMode: displayMode,
        throwOnError: false,
        strict: 'ignore',
        trust: false
      });
      node.dataset.rendered = '1';
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderAll);
  } else {
    renderAll();
  }
})();
