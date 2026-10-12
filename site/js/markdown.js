/* ============================================================
 * 轻量 Markdown 渲染器（为本书内容定制）
 * 支持：标题/段落/代码块(带语言)/表格/列表(嵌套)/引用/分隔线
 *       行内：加粗/行内代码/链接；输出 HTML
 * ============================================================ */
(function (global) {
  'use strict';

  function escapeHtml(s) {
    return String(s)
      .replace(/&(?!(amp|lt|gt|quot|#\d+);)/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function slugify(text) {
    return 'sec-' + text
      .replace(/[^\w\u4e00-\u9fa5]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 60);
  }

  /* 行内解析：`code`、**bold**、[text](url) */
  function inline(s) {
    s = escapeHtml(s);
    // 行内代码
    s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
    // 链接 [text](url)
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    // 加粗 **text**
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    return s;
  }

  /* 代码块高亮（委托给 highlight.js） */
  function codeBlock(lang, code) {
    var hl = (global.Highlighter && global.Highlighter.highlight) ? global.Highlighter.highlight(code, lang) : escapeHtml(code);
    var langLabel = lang ? '<span class="code-lang">' + escapeHtml(lang) + '</span>' : '';
    return '<pre><button class="code-copy" data-copy>复制</button>' + langLabel + '<code>' + hl + '</code></pre>';
  }

  /* 表格行解析 */
  function parseTableRow(line) {
    var cells = line.trim().replace(/^\||\|$/g, '').split('|');
    return cells.map(function (c) { return c.trim(); });
  }

  function renderTable(rows) {
    var head = rows[0];
    var body = rows.slice(2); // 第 2 行是分隔行
    var html = '<table><thead><tr>';
    head.forEach(function (c) { html += '<th>' + inline(c) + '</th>'; });
    html += '</tr></thead><tbody>';
    body.forEach(function (row) {
      html += '<tr>';
      head.forEach(function (_, i) { html += '<td>' + inline(row[i] || '') + '</td>'; });
      html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
  }

  /* 列表块：items 为 [{depth, ordered, text}] */
  function renderList(items) {
    var html = '';
    var stack = []; // {ordered, depth}
    items.forEach(function (it) {
      // 关闭比当前更深、或同层但列表类型不同的层级（栈帧表示"打开的列表"）
      while (stack.length && (stack[stack.length - 1].depth > it.depth ||
             (stack[stack.length - 1].depth === it.depth && stack[stack.length - 1].ordered !== it.ordered))) {
        var closed = stack.pop();
        html += '</li></' + (closed.ordered ? 'ol' : 'ul') + '>';
      }
      if (stack.length && stack[stack.length - 1].depth === it.depth &&
          stack[stack.length - 1].ordered === it.ordered) {
        html += '</li>'; // 同级同类型：关闭上一个 li，继续同一列表（不入栈）
      } else {
        html += '<' + (it.ordered ? 'ol' : 'ul') + '>'; // 新列表或更深层级
        stack.push({ depth: it.depth, ordered: it.ordered });
      }
      html += '<li>' + inline(it.text);
    });
    while (stack.length) {
      var closed = stack.pop();
      html += '</li></' + (closed.ordered ? 'ol' : 'ul') + '>';
    }
    return html;
  }

  /* 主渲染：markdown → HTML */
  function render(md) {
    if (!md) return '';
    var lines = String(md).split('\n');
    var html = '';
    var i = 0;
    var headingIds = [];

    function flushList() {
      if (listBuf.length) { html += renderList(listBuf); listBuf = []; }
    }

    var listBuf = [];

    while (i < lines.length) {
      var line = lines[i];

      // 代码块（围栏式）。支持「嵌套围栏」写法：正文里用外层 text 包裹一段
      // ```mermaid 示例时，内部的 info-string 围栏会被当作内容，直到外层围栏闭合。
      // 判定规则：遇到 info-string 围栏视为在「文档层」开始一个块；块内再遇到
      // info-string 围栏先记为「嵌套开启」，遇到裸围栏优先闭合最近一次开启。
      var fenceOpen = line.match(/^(`{3,}|~{3,})[ \t]*([\w-]*)[ \t]*$/);
      if (fenceOpen) {
        flushList();
        var fenceSeq = fenceOpen[1];
        var fenceChar = fenceSeq.charAt(0);
        var fenceLen = fenceSeq.length;
        var outerLang = fenceOpen[2] || '';
        var buf = [];
        i++;
        var nested = 0; // 块内嵌套开启计数（未闭合的 info-string 围栏）
        var bareRe = new RegExp('^' + (fenceChar === '`' ? '`' : '~') + '{' + fenceLen + ',}[ \t]*$');
        var anyFenceRe = new RegExp('^' + (fenceChar === '`' ? '`' : '~') + '{' + fenceLen + ',}[ \\t]*([\\w-]*)[ \\t]*$');
        while (i < lines.length) {
          var cur = lines[i];
          if (bareRe.test(cur)) {
            if (nested > 0) { nested--; buf.push(cur); i++; continue; } // 先闭合内层
            break; // 闭合外层
          }
          var am = cur.match(anyFenceRe);
          if (am && am[1]) { nested++; buf.push(cur); i++; continue; } // 内层 info-string 围栏开启
          buf.push(cur); i++;
        }
        i++; // 跳过外层结束围栏
        html += codeBlock(outerLang, buf.join('\n'));
        continue;
      }

      // 标题
      var hMatch = line.match(/^(#{1,4})\s+(.*)$/);
      if (hMatch) {
        flushList();
        var level = hMatch[1].length;
        var text = hMatch[2];
        var id = slugify(text);
        // 去掉标题里的 ★ 频率标注，避免锚点过长
        var cleanText = text.replace(/\s*★+.*$/, '').trim();
        if (level === 2) headingIds.push({ id: slugify(cleanText), title: cleanText });
        html += '<h' + level + ' id="' + slugify(cleanText) + '" class="anchor-target">' + inline(cleanText) + '</h' + level + '>';
        i++;
        continue;
      }

      // 表格（当前行以 | 开头且下一行是分隔行）
      if (/^\|.*\|$/.test(line.trim()) && i + 1 < lines.length && /^\|?[\s:|-]*-+[\s:|-]*\|?$/.test(lines[i + 1])) {
        flushList();
        var tRows = [];
        while (i < lines.length && /^\|.*\|$/.test(lines[i].trim())) { tRows.push(lines[i]); i++; }
        html += renderTable(tRows.map(parseTableRow));
        continue;
      }

      // 引用块
      if (/^>\s?/.test(line)) {
        flushList();
        var quote = [];
        while (i < lines.length && /^>\s?/.test(lines[i])) { quote.push(lines[i].replace(/^>\s?/, '')); i++; }
        html += '<blockquote>' + quote.map(function (q) { return '<p>' + inline(q) + '</p>'; }).join('') + '</blockquote>';
        continue;
      }

      // 列表
      var listMatch = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
      if (listMatch) {
        var depth = Math.floor(listMatch[1].length / 2);
        var ordered = /^\d+\.$/.test(listMatch[2]);
        listBuf.push({ depth: depth, ordered: ordered, text: listMatch[3] });
        i++;
        continue;
      }

      // 分隔线
      if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
        flushList();
        html += '<hr>';
        i++;
        continue;
      }

      // 空行
      if (/^\s*$/.test(line)) {
        flushList();
        i++;
        continue;
      }

      // 普通段落（支持合并多行，遇到块级元素停止）
      flushList();
      var para = [line];
      i++;
      while (i < lines.length) {
        var nl = lines[i];
        if (/^\s*$/.test(nl) || /^```/.test(nl) || /^#{1,4}\s/.test(nl) || /^>\s?/.test(nl) ||
            /^\|.*\|$/.test(nl.trim()) || /^(\s*)([-*]|\d+\.)\s+/.test(nl) || /^\s*(-{3,}|\*{3,})\s*$/.test(nl)) {
          break;
        }
        para.push(nl);
        i++;
      }
      html += '<p>' + para.map(inline).join('<br>') + '</p>';
    }
    flushList();
    return { html: html, headings: headingIds };
  }

  global.Markdown = { render: render };
})(window);
