/*
 * Read-aloud for the curriculum pages.
 *
 * Uses the browser's built-in speechSynthesis: no dependency, no CDN, no API key, works
 * offline. Pre-generated audio would sound better but would put unreviewable binaries in
 * the repo that need regenerating on every content edit - the same reason the
 * infographics are SVG rather than GIF.
 *
 * The hard part is not playback, it is making technical prose listenable:
 *
 *   - Code blocks are skipped, not spelled out. A synthesiser reading
 *     `hipMalloc(&d_A, bytes)` aloud is worse than silence, so blocks are announced and
 *     stepped over.
 *   - SAY_AS rewrites the vocabulary this curriculum actually uses, so "TFLOP/s" is read
 *     "teraflops per second" rather than "T-F-L-O-P slash S", and "gfx942" is spelled out.
 *   - Collapsed <details> are skipped. Section 9 hides answers behind them deliberately;
 *     reading them aloud would spoil the retrieval practice.
 *   - Highlighting is per block, not per word: no DOM surgery on the page's own markup,
 *     and nothing reflows mid-sentence.
 *
 * This is a convenience feature, NOT an accessibility solution. Real accessibility is the
 * semantic HTML underneath; a screen reader will always do this better.
 */
(function () {
  "use strict";

  if (!("speechSynthesis" in window) || !("SpeechSynthesisUtterance" in window)) {
    return;                       // no dead UI on browsers that cannot do this
  }

  var PREFS = "ppc.readaloud.v1";

  /* Spoken forms for the vocabulary in these modules. Order matters: longest and most
     specific first, because these run as a sequence of replacements. */
  var SAY_AS = [
    [/\bTFLOP\/s\b/g, "teraflops per second"],
    [/\bGFLOP\/s\b/g, "gigaflops per second"],
    [/\bPFLOP\/s\b/g, "petaflops per second"],
    [/\bFLOP\/byte\b/g, "flops per byte"],
    [/\bFLOPs\b/g, "flops"],
    [/\bTB\/s\b/g, "terabytes per second"],
    [/\bGB\/s\b/g, "gigabytes per second"],
    [/\bMB\/s\b/g, "megabytes per second"],
    [/\bGiB\b/g, "gibibytes"], [/\bMiB\b/g, "mebibytes"], [/\bKiB\b/g, "kibibytes"],
    [/µs\b/g, " microseconds"], [/\bms\b/g, "milliseconds"], [/\bns\b/g, "nanoseconds"],
    [/\bbf16\b/gi, "B-float 16"],
    [/\bfp(\d+)\b/gi, "F P $1"],
    [/\bint(\d+)\b/gi, "int $1"],
    [/\bgfx(\w+)\b/gi, function (_, d) { return "G F X " + d.split("").join(" "); }],
    [/\bsm_(\d+)\b/gi, "S M $1"],
    [/\bKV\b/g, "K V"],
    [/\bO\(([^)]{1,24})\)/g, "big O of $1"],
    [/\bvs\.?(?=\s)/gi, "versus"],
    [/\be\.g\.\s*/gi, "for example, "],
    [/\bi\.e\.\s*/gi, "that is, "],
    [/\betc\./gi, "etcetera"],
    [/\u00d7/g, " times "],           //  x
    [/\u2192/g, " to "],              //  ->
    [/\u2248/g, " approximately "],   //  ~=
    [/\u2264/g, " at most "],         //  <=
    [/\u2265/g, " at least "],        //  >=
    [/\u00b2/g, " squared"],
    [/\u00b3/g, " cubed"],
    [/\u00b7/g, ", "],                //  middot
    [/[\u2014\u2013]/g, ", "],        //  em / en dash -> a breath, not a word
    [/`/g, ""],
    [/_/g, " "],
    [/\s{2,}/g, " "]
  ];

  function speakable(text) {
    var out = text;
    SAY_AS.forEach(function (r) { out = out.replace(r[0], r[1]); });
    return out.trim();
  }

  /* Skip anything that would be nonsense or a spoiler when spoken. */
  function skippable(el) {
    if (el.closest("pre, table, .mermaid, .highlight, .ra-player, .md-source-file, .md-nav, .md-footer")) {
      return true;
    }
    var d = el.closest("details");
    return !!(d && !d.open);        // respect a collapsed answers block
  }

  function collect(article) {
    var units = [];
    var nodes = article.querySelectorAll("h1, h2, h3, p, li, blockquote");
    Array.prototype.forEach.call(nodes, function (el) {
      if (skippable(el)) { return; }
      if (el.tagName === "BLOCKQUOTE" || el.tagName === "LI") {
        // Only take these when they hold text directly; otherwise their child <p>
        // elements are already in the list and we would read them twice.
        if (el.querySelector("p, li")) { return; }
      }
      var raw = (el.textContent || "").replace(/\s+/g, " ").trim();
      if (raw.length < 2) { return; }

      var heading = /^H[123]$/.test(el.tagName);
      var text = speakable(raw);
      if (heading) { text = "Section. " + text + "."; }
      if (!text) { return; }

      units.push({ el: el, text: text, heading: heading, label: raw.slice(0, 60) });
    });
    return units;
  }

  // ---- state ---------------------------------------------------------------------
  var units = [], i = 0, playing = false, currentEl = null;
  var prefs = (function () {
    try { return JSON.parse(localStorage.getItem(PREFS)) || {}; } catch (e) { return {}; }
  })();
  var rate = prefs.rate || 1;
  var voiceURI = prefs.voice || "";
  var player, whereEl, playBtn;

  function savePrefs() {
    try { localStorage.setItem(PREFS, JSON.stringify({ rate: rate, voice: voiceURI })); }
    catch (e) { /* private mode */ }
  }

  function voices() {
    return window.speechSynthesis.getVoices().filter(function (v) { return /^en/i.test(v.lang); });
  }

  function chosenVoice() {
    var all = voices();
    return all.filter(function (v) { return v.voiceURI === voiceURI; })[0] || null;
  }

  function highlight(el) {
    if (currentEl) { currentEl.classList.remove("ra-now"); }
    currentEl = el || null;
    if (!currentEl) { return; }
    currentEl.classList.add("ra-now");
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    currentEl.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
  }

  function speakCurrent() {
    if (i >= units.length) { return stop(); }
    var u = units[i];
    highlight(u.el);
    whereEl.textContent = u.label;

    var utt = new SpeechSynthesisUtterance(u.text);
    utt.rate = rate;
    var v = chosenVoice();
    if (v) { utt.voice = v; utt.lang = v.lang; }
    utt.onend = function () {
      if (!playing) { return; }
      i += 1;
      speakCurrent();
    };
    // A synthesis error should stop cleanly rather than wedge the player.
    utt.onerror = function (e) { if (e && e.error === "interrupted") { return; } stop(); };
    window.speechSynthesis.speak(utt);
  }

  function play() {
    if (!units.length) { return; }
    playing = true;
    player.hidden = false;
    playBtn.textContent = "Pause";
    playBtn.setAttribute("aria-label", "Pause reading");
    window.speechSynthesis.cancel();
    speakCurrent();
  }

  function pause() {
    playing = false;
    window.speechSynthesis.cancel();   // pause/resume is unreliable across engines
    playBtn.textContent = "Play";
    playBtn.setAttribute("aria-label", "Resume reading");
  }

  function stop() {
    playing = false;
    window.speechSynthesis.cancel();
    highlight(null);
    player.hidden = true;
    playBtn.textContent = "Play";
    i = 0;
  }

  function jump(delta) {
    // Move by heading so the controls step through sections, not paragraphs.
    var target = i;
    var step = delta > 0 ? 1 : -1;
    for (var k = i + step; k >= 0 && k < units.length; k += step) {
      if (units[k].heading) { target = k; break; }
      if (k === 0) { target = 0; }
    }
    i = Math.max(0, Math.min(units.length - 1, target));
    if (playing) { window.speechSynthesis.cancel(); speakCurrent(); }
    else { highlight(units[i].el); whereEl.textContent = units[i].label; }
  }

  // ---- UI ------------------------------------------------------------------------
  function buildPlayer() {
    player = document.createElement("div");
    player.className = "ra-player";
    player.setAttribute("role", "region");
    player.setAttribute("aria-label", "Read aloud controls");
    player.hidden = true;

    player.innerHTML =
      '<button type="button" data-ra="prev" aria-label="Previous section">Prev</button>' +
      '<button type="button" data-ra="play" class="ra-primary">Pause</button>' +
      '<button type="button" data-ra="next" aria-label="Next section">Next</button>' +
      '<button type="button" data-ra="stop" aria-label="Stop reading">Stop</button>' +
      '<span class="ra-sep"></span>' +
      '<label class="ra-where" id="ra-where">&nbsp;</label>' +
      '<span class="ra-sep"></span>' +
      '<select data-ra="rate" aria-label="Speaking rate">' +
        ["0.8", "1", "1.25", "1.5", "1.75", "2"].map(function (r) {
          return '<option value="' + r + '"' + (String(rate) === r ? " selected" : "") + ">" + r + "x</option>";
        }).join("") +
      "</select>" +
      '<select data-ra="voice" aria-label="Voice"></select>';

    document.body.appendChild(player);
    whereEl = player.querySelector("#ra-where");
    playBtn = player.querySelector('[data-ra="play"]');

    fillVoices();
    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = fillVoices;
    }

    player.addEventListener("click", function (e) {
      var b = e.target.closest("button[data-ra]");
      if (!b) { return; }
      var a = b.getAttribute("data-ra");
      if (a === "play") { playing ? pause() : play(); }
      else if (a === "stop") { stop(); }
      else if (a === "next") { jump(1); }
      else if (a === "prev") { jump(-1); }
    });

    player.addEventListener("change", function (e) {
      var sel = e.target;
      if (sel.getAttribute("data-ra") === "rate") {
        rate = parseFloat(sel.value); savePrefs();
        if (playing) { window.speechSynthesis.cancel(); speakCurrent(); }
      } else if (sel.getAttribute("data-ra") === "voice") {
        voiceURI = sel.value; savePrefs();
        if (playing) { window.speechSynthesis.cancel(); speakCurrent(); }
      }
    });
  }

  function fillVoices() {
    var sel = player.querySelector('[data-ra="voice"]');
    var list = voices();
    if (!list.length) { sel.innerHTML = '<option value="">Default voice</option>'; return; }
    sel.innerHTML = list.map(function (v) {
      return '<option value="' + v.voiceURI + '"' + (v.voiceURI === voiceURI ? " selected" : "") +
             ">" + v.name.replace(/\s*\(.*\)$/, "") + "</option>";
    }).join("");
  }

  function buildStartButton(article, count) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ra-start";
    btn.innerHTML =
      '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 6v4h2.5L9 13V3L5.5 6H3zm8.5 2a3 3 0 0 0-1.5-2.6v5.2A3 3 0 0 0 11.5 8z"/></svg>' +
      "Listen to this page";
    btn.setAttribute("aria-label", "Read this page aloud, " + count + " passages");

    btn.addEventListener("click", function () {
      if (playing) { pause(); } else { i = 0; play(); }
    });

    var first = article.querySelector("h1");
    if (first && first.nextSibling) { first.parentNode.insertBefore(btn, first.nextSibling); }
    else { article.insertBefore(btn, article.firstChild); }
  }

  function init() {
    var article = document.querySelector(".md-content__inner") || document.querySelector("article");
    if (!article) { return; }

    units = collect(article);
    if (units.length < 3) { return; }   // not enough prose to be worth offering

    buildPlayer();
    buildStartButton(article, units.length);

    // Leaving the page mid-sentence should not keep the synthesiser talking.
    window.addEventListener("beforeunload", function () { window.speechSynthesis.cancel(); });
    window.addEventListener("pagehide", function () { window.speechSynthesis.cancel(); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
