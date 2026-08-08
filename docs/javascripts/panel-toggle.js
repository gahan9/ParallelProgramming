/*
 * Header buttons that hide/show the left navigation rail (the full track/module tree)
 * and the right "On this page" table of contents.
 *
 * Neither panel is wrong to have - they are how a reader gets around a 24-module site -
 * but a module page can carry 30+ sub-headings in its TOC and a track tree with every
 * other module in it, which is a lot of chrome while you are just trying to read one
 * section. This makes both optional per reader, remembered across pages via
 * localStorage, without touching any page content or Material's own responsive
 * behavior (below the desktop breakpoint both panels already collapse into the mobile
 * drawer / inline TOC, so the toggle hides itself there instead of controlling nothing).
 */
(function () {
  "use strict";

  var KEY = "ppc.panels.v1";
  var state = (function () {
    try { return JSON.parse(localStorage.getItem(KEY)) || {}; }
    catch (e) { return {}; }
  })();

  function apply() {
    document.documentElement.classList.toggle("ppc-nav-hidden", !!state.nav);
    document.documentElement.classList.toggle("ppc-toc-hidden", !!state.toc);
  }
  apply();   // run before building the buttons so a returning reader sees no flash-open

  function save() {
    try { localStorage.setItem(KEY, JSON.stringify(state)); }
    catch (e) { /* private mode */ }
  }

  function describe(label, hidden) {
    return (hidden ? "Show " : "Hide ") + label.toLowerCase();
  }

  function makeButton(label, key, lineX) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ppc-panel-btn";

    function refresh() {
      var hidden = !!state[key];
      btn.setAttribute("aria-pressed", String(hidden));
      btn.title = describe(label, hidden);
      btn.setAttribute("aria-label", describe(label, hidden));
    }

    btn.innerHTML =
      '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
        '<rect x="3.5" y="4.5" width="17" height="15" rx="2"></rect>' +
        '<line x1="' + lineX + '" y1="4.5" x2="' + lineX + '" y2="19.5"></line>' +
      "</svg>" +
      '<span class="ppc-panel-btn__label">' + label + "</span>";
    refresh();

    btn.addEventListener("click", function () {
      state[key] = !state[key];
      save();
      apply();
      refresh();
    });

    return btn;
  }

  function init() {
    var header = document.querySelector(".md-header__inner");
    if (!header || header.querySelector(".ppc-panel-toggles")) { return; }

    var wrap = document.createElement("div");
    wrap.className = "ppc-panel-toggles";
    wrap.appendChild(makeButton("Menu", "nav", 9));
    wrap.appendChild(makeButton("Contents", "toc", 15));
    header.appendChild(wrap);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
