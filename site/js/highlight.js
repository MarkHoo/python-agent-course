/* ============================================================
 * 轻量代码高亮器（自包含、无外部依赖）
 * 支持：注释、字符串、关键字、数字、XML 标签/属性
 * ============================================================ */
(function (global) {
  'use strict';

  var KEYWORDS = [
    'def', 'import', 'return', 'if', 'else', 'for', 'while', 'do', 'break', 'continue',
    'class', 'new', 'null', 'true', 'false', 'void', 'this', 'super', 'extends', 'implements',
    'public', 'private', 'protected', 'static', 'final', 'try', 'catch', 'finally', 'throw',
    'switch', 'case', 'default', 'instanceof', 'var', 'let', 'const', 'function', 'async', 'await',
    'in', 'of', 'not', 'and', 'or', 'then', 'end', 'begin', 'require'
  ].sort(function (a, b) { return b.length - a.length; }).join('|');

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  var TOKEN_RE = new RegExp(
    '("(?:[^"\\\\]|\\\\.)*"|\'(?:[^\'\\\\]|\\\\.)*\')' +            // 1 字符串
    '|(\\/\\/[^\\n]*|#[^\\n]*|--[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/)' +   // 2 注释
    '|(\\b(?:' + KEYWORDS + ')\\b)' +                                 // 3 关键字
    '|(\\b\\d+(?:\\.\\d+)?\\b)' +                                      // 4 数字
    '|(<\\/?[a-zA-Z][^>]*>|"[^"]*"\\s*=)' +                           // 5 XML 标签/属性
    '|([$]\\{?[A-Za-z_][\\w.]*\\}?)',                                 // 6 变量 ${xxx}
    'g'
  );

  function highlight(code, lang) {
    var s = String(code);
    return s.replace(TOKEN_RE, function (m, str, comment, kw, num, tag, variable) {
      if (str !== undefined) return '<span class="hl-string">' + escapeHtml(str) + '</span>';
      if (comment !== undefined) return '<span class="hl-comment">' + escapeHtml(comment) + '</span>';
      if (kw !== undefined) return '<span class="hl-keyword">' + escapeHtml(kw) + '</span>';
      if (num !== undefined) return '<span class="hl-number">' + escapeHtml(num) + '</span>';
      if (tag !== undefined) {
        if (/^<[^>]*>$/.test(tag)) return '<span class="hl-tag">' + escapeHtml(tag) + '</span>';
        return '<span class="hl-attr">' + escapeHtml(tag) + '</span>';
      }
      if (variable !== undefined) return '<span class="hl-func">' + escapeHtml(variable) + '</span>';
      return escapeHtml(m);
    });
  }

  global.Highlighter = { highlight: highlight };
})(window);
