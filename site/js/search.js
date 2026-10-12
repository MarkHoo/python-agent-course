/* ============================================================
 * 全文搜索：对全书数据建立索引，支持空格分隔多关键词（AND）
 * ============================================================ */
(function (global) {
  'use strict';

  var index = [];

  function plainText(md) {
    return String(md)
      .replace(/```[\s\S]*?```/g, ' ')          // 代码块整体保留为占位（搜索仍可命中标题）
      .replace(/^#{1,4}\s+/gm, '')
      .replace(/^>\s?/gm, '')
      .replace(/^\s*\|/gm, ' ').replace(/\|\s*$/gm, ' ')
      .replace(/^\s*[-*]\s+/gm, ' ')
      .replace(/^\s*\d+\.\s+/gm, ' ')
      .replace(/[*`~]/g, '')
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function buildIndex() {
    index = [];
    var keys = Object.keys(global.BOOK_DATA || {});
    keys.forEach(function (key) {
      var md = global.BOOK_DATA[key];
      var titleMatch = md.match(/^#\s+(.+)$/m);
      var title = titleMatch ? titleMatch[1] : key;
      var text = plainText(md);
      index.push({ key: key, title: title, text: text });
    });
  }

  function highlightSnippet(text, words) {
    var lower = text.toLowerCase();
    var firstPos = -1;
    words.forEach(function (w) {
      var p = lower.indexOf(w);
      if (p >= 0 && (firstPos < 0 || p < firstPos)) firstPos = p;
    });
    if (firstPos < 0) firstPos = 0;
    var start = Math.max(0, firstPos - 50);
    var snippet = (start > 0 ? '…' : '') + text.slice(start, start + 160) + (start + 160 < text.length ? '…' : '');
    words.forEach(function (w) {
      var re = new RegExp('(' + w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
      snippet = snippet.replace(re, '<mark>$1</mark>');
    });
    return snippet;
  }

  function search(query) {
    if (!query || !query.trim()) return [];
    var words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    var results = [];
    index.forEach(function (item) {
      var titleHit = words.every(function (w) { return item.title.toLowerCase().indexOf(w) >= 0; });
      var textHit = words.every(function (w) { return item.text.toLowerCase().indexOf(w) >= 0; });
      if (titleHit || textHit) {
        results.push({
          key: item.key,
          title: item.title,
          snippet: titleHit ? '【标题命中】' + highlightSnippet(item.title, words) : highlightSnippet(item.text, words),
          score: (titleHit ? 10 : 0) + (textHit ? 1 : 0)
        });
      }
    });
    results.sort(function (a, b) { return b.score - a.score; });
    return results;
  }

  global.Search = { buildIndex: buildIndex, search: search, getIndex: function () { return index; } };
})(window);
