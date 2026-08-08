/*
 * Upgrades GitHub-style alert blockquotes (`> [!NOTE]`, `[!TIP]`, `[!IMPORTANT]`,
 * `[!WARNING]`, `[!CAUTION]`) into real Material admonitions, client-side.
 *
 * GitHub renders that syntax natively; MkDocs Material does not (this repo enables
 * `admonition` + `pymdownx.details`, not a GitHub-alert extension), so on the docs site
 * it was falling through to a plain blockquote with a literal "[!NOTE]" line - readable
 * on GitHub, broken here. Thirteen module READMEs already use the syntax because it is
 * the correct choice for the repo view. Rewriting hand-authored content to satisfy one
 * renderer while breaking the other is the wrong trade; this makes both renderers correct.
 */
(function () {
  "use strict";

  var ALERTS = {
    NOTE:      { cls: "note",    title: "Note" },
    TIP:       { cls: "tip",     title: "Tip" },
    IMPORTANT: { cls: "info",    title: "Important" },
    WARNING:   { cls: "warning", title: "Warning" },
    CAUTION:   { cls: "danger",  title: "Caution" }
  };
  var MARKER = /^\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*/;

  function upgrade(bq) {
    var first = bq.firstElementChild;
    if (!first || first.tagName !== "P") { return; }

    var lead = first.firstChild;
    if (!lead || lead.nodeType !== Node.TEXT_NODE) { return; }

    var m = MARKER.exec(lead.textContent);
    if (!m) { return; }
    var alert = ALERTS[m[1]];

    var rest = lead.textContent.slice(m[0].length);
    if (rest) { lead.textContent = rest; }
    else { first.removeChild(lead); }

    bq.classList.add("admonition", alert.cls);
    var title = document.createElement("p");
    title.className = "admonition-title";
    title.textContent = alert.title;
    bq.insertBefore(title, first);

    // Marker-only paragraph (e.g. followed by a heading on its own line): drop the now
    // empty <p> rather than leave a blank line inside the admonition.
    if (!first.textContent.trim() && !first.querySelector("*")) {
      bq.removeChild(first);
    }
  }

  function init() {
    var blockquotes = document.querySelectorAll(".md-typeset blockquote");
    Array.prototype.forEach.call(blockquotes, upgrade);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
