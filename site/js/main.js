/* ============================================================
 * 主应用逻辑：目录、渲染、导航、搜索联动、进度、主题、复制等
 * ============================================================ */
(function () {
  'use strict';

  var $ = function (sel) { return document.querySelector(sel); };
  var $$ = function (sel) { return Array.prototype.slice.call(document.querySelectorAll(sel)); };

  // ---------- 章节顺序（以数据加载顺序为准） ----------
  var keys = Object.keys(window.BOOK_DATA || []);
  var chapterMeta = keys.map(function (key) {
    var md = window.BOOK_DATA[key];
    var titleMatch = md.match(/^#\s+(.+)$/m);
    var title = titleMatch ? titleMatch[1].replace(/\s*★+.*$/, '') : key;
    // 提取 ## 小节标题
    var headings = [];
    var re = /^##\s+(.+)$/gm, m;
    while ((m = re.exec(md)) !== null) {
      headings.push({ title: m[1].replace(/\s*★+.*$/, '').trim() });
    }
    return { key: key, title: title, headings: headings };
  });

  var currentKey = null;
  var lastQuery = '';

  // ---------- 目录树 ----------
  function renderToc() {
    var ul = $('#tocList');
    ul.innerHTML = '';
    chapterMeta.forEach(function (meta, idx) {
      var li = document.createElement('li');
      var btn = document.createElement('button');
      btn.className = 'toc-chapter';
      btn.innerHTML = '<span class="toc-num">' + shortNum(idx) + '</span><span class="toc-name">' + escapeText(meta.title) + '</span>' +
        (meta.headings.length ? '<span class="toc-toggle">▾</span>' : '');
      btn.addEventListener('click', function () {
        if (currentKey === meta.key) {
          ul2.classList.toggle('open'); // 已在当前章节：手动折叠/展开
        } else {
          go(meta.key); // 跳转章节（go 内自动展开其子列表）
        }
      });
      li.appendChild(btn);
      if (meta.headings.length) {
        var ul2 = document.createElement('ul');
        ul2.className = 'toc-subs';
        ul2.setAttribute('data-key', meta.key);
        meta.headings.forEach(function (h) {
          var li2 = document.createElement('li');
          var b2 = document.createElement('button');
          b2.className = 'toc-section';
          b2.textContent = h.title;
          b2.addEventListener('click', function () {
            go(meta.key, h.title);
            closeSidebar();
          });
          li2.appendChild(b2);
          ul2.appendChild(li2);
        });
        li.appendChild(ul2);
      }
      ul.appendChild(li);
    });
  }

  function shortNum(idx) {
    var k = chapterMeta[idx].key;
    if (k === 'README') return '序';
    if (k === 'HANDOVER') return '移';
    if (k === 'CHAPTER_GUIDE') return '规';
    if (k === 'ENGINEERING_BASELINE') return '基';
    if (k === 'TROUBLESHOOTING') return '障';
    if (k === 'CONTRIBUTING') return '贡';
    var m = k.match(/^ch(\d+)/);
    if (m) return String(parseInt(m[1], 10));
    return '·';
  }

  function escapeText(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ---------- 首页目录卡片 ----------
  function renderHome() {
    var home = $('#homeToc');
    home.innerHTML = '';
    // 分组
    var groups = [
      { label: '📖 入门与转型（第 1~3 章）', keys: ['ch01', 'ch02', 'ch03'] },
      { label: '🔧 Tool · MCP · Agent 循环（第 4~7 章）', keys: ['ch04', 'ch05', 'ch06', 'ch07'] },
      { label: '🧠 RAG 基础：向量 · 切分 · Milvus（第 8~12 章）', keys: ['ch08', 'ch09', 'ch10', 'ch11', 'ch12'] },
      { label: '💬 上下文与结构化输出（第 13~15 章）', keys: ['ch13', 'ch14', 'ch15'] },
      { label: '⛓️ LCEL · Prompt · Runnable（第 16~19 章）', keys: ['ch16', 'ch17', 'ch18', 'ch19'] },
      { label: '🌊 流式 · 定时 · 语音 · 图编排（第 20~26 章）', keys: ['ch20', 'ch21', 'ch22', 'ch23', 'ch24', 'ch25', 'ch26'] },
      { label: '🚀 工程化与存储（第 27~43 章）', keys: ['ch27', 'ch28', 'ch29', 'ch30', 'ch31', 'ch32', 'ch33', 'ch34', 'ch35', 'ch36', 'ch37', 'ch38', 'ch39', 'ch40', 'ch41', 'ch42', 'ch43'] },
      { label: '🎯 实战项目：简历库 · 面试 · 企业知识库（第 44~58 章）', keys: ['ch44', 'ch45', 'ch46', 'ch47', 'ch48', 'ch49', 'ch50', 'ch51', 'ch52', 'ch53', 'ch54', 'ch55', 'ch56', 'ch57', 'ch58'] },
      { label: '📄 序言与阅读指南', keys: ['README'] },
      { label: '📐 工程交接与规范', keys: ['HANDOVER', 'CHAPTER_GUIDE', 'ENGINEERING_BASELINE', 'TROUBLESHOOTING', 'CONTRIBUTING'] }
    ];
    groups.forEach(function (g) {
      var card = document.createElement('div');
      card.className = 'home-group';
      var h = document.createElement('h3');
      h.textContent = g.label;
      card.appendChild(h);
      var list = document.createElement('div');
      list.className = 'home-group-list';
      chapterMeta.forEach(function (meta) {
        if (g.keys.some(function (p) { return meta.key.indexOf(p) === 0; })) {
          var b = document.createElement('button');
          b.className = 'home-chapter';
          b.textContent = meta.title;
          b.addEventListener('click', function () { go(meta.key); });
          list.appendChild(b);
        }
      });
      card.appendChild(list);
      home.appendChild(card);
    });
  }

  // ---------- 渲染章节 ----------
  function go(key, sectionTitle) {
    currentKey = key;
    var md = window.BOOK_DATA[key];
    var rendered = window.Markdown.render(md);
    var article = $('#articleView');
    var content = $('#articleContent');
    content.innerHTML = rendered.html;
    var meta = chapterMeta.filter(function (x) { return x.key === key; })[0] || { title: key };

    // 上/下一章
    var idx = chapterMeta.findIndex(function (x) { return x.key === key; });
    var prev = idx > 0 ? chapterMeta[idx - 1] : null;
    var next = idx < chapterMeta.length - 1 ? chapterMeta[idx + 1] : null;
    $('#btnPrev').disabled = !prev;
    $('#btnNext').disabled = !next;
    $('#btnPrev').innerHTML = prev ? '← ' + escapeText(prev.title) : '←';
    $('#btnNext').innerHTML = next ? escapeText(next.title) + ' →' : '→';
    $('#btnPrev').onclick = function () { if (prev) go(prev.key); };
    $('#btnNext').onclick = function () { if (next) go(next.key); };

    // 目录高亮与展开：当前章节子列表展开，其余折叠；小节定位时高亮对应小节
    $$('.toc-chapter').forEach(function (b) {
      b.classList.toggle('active', b.textContent.indexOf(meta.title) >= 0);
    });
    $$('.toc-subs').forEach(function (u) { u.classList.remove('open'); });
    var curSub = document.querySelector('.toc-subs[data-key="' + key + '"]');
    if (curSub) curSub.classList.add('open');
    $$('.toc-section').forEach(function (b) { b.classList.remove('active'); });
    if (sectionTitle) {
      var cleanTitle = sectionTitle.replace(/\s*★+.*$/, '').trim();
      $$('.toc-section').forEach(function (b) {
        if (b.textContent.trim() === cleanTitle) b.classList.add('active');
      });
    }

    // URL hash
    try { history.replaceState(null, '', '#/ch/' + encodeURIComponent(key)); } catch (e) { location.hash = '#/ch/' + encodeURIComponent(key); }

    // 显示章节视图
    $('#homeView').classList.add('hidden');
    article.classList.remove('hidden');

    // 滚动定位：小节定位滚动到锚点（不重置顶部）；章节/搜索跳转回顶部
    if (sectionTitle && !lastQuery) {
      scrollToSection(sectionTitle);
    } else {
      if (lastQuery) highlightMatches(lastQuery);
      window.scrollTo(0, 0);
    }
    closeSidebar();
  }

  function scrollToSection(title) {
    var clean = title.replace(/\s*★+.*$/, '').trim();
    var slug = 'sec-' + clean.replace(/[^\w\u4e00-\u9fa5]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);
    var el = document.getElementById(slug);
    // 兜底：按文本模糊匹配 h2/h3
    if (!el) {
      $$('.markdown-body h2, .markdown-body h3').some(function (h) {
        if (h.textContent.replace(/\s+/g, '') === clean.replace(/\s+/g, '')) { el = h; return true; }
        return false;
      });
    }
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ---------- 搜索词在正文中高亮 ----------
  function highlightMatches(query) {
    var words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!words.length) return;
    var walker = document.createTreeWalker($('#articleContent'), NodeFilter.SHOW_TEXT, null);
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(function (node) {
      var lower = node.nodeValue.toLowerCase();
      var hit = words.some(function (w) { return lower.indexOf(w) >= 0; });
      if (hit && node.parentNode && node.parentNode.className !== 'hl-string') {
        var span = document.createElement('mark');
        span.textContent = node.nodeValue;
        node.parentNode.replaceChild(span, node);
      }
    });
  }

  // ---------- 首页 ----------
  function showHome() {
    currentKey = null;
    $('#articleView').classList.add('hidden');
    $('#homeView').classList.remove('hidden');
    $$('.toc-chapter').forEach(function (b) { b.classList.remove('active'); });
    try { history.replaceState(null, '', '/'); } catch (e) { try { location.hash = '#home'; } catch (e2) {} }
    window.scrollTo(0, 0);
  }

  // ---------- 搜索 ----------
  function bindSearch() {
    var input = $('#searchInput');
    var panel = $('#searchPanel');
    var results = $('#searchResults');
    var empty = $('#searchEmpty');
    var debounceTimer = null;

    function doSearch() {
      var q = input.value.trim();
      lastQuery = q;
      var res = window.Search.search(q);
      results.innerHTML = '';
      empty.classList.add('hidden');
      if (!q) { panel.classList.add('hidden'); return; }
      if (!res.length) { empty.classList.remove('hidden'); panel.classList.remove('hidden'); return; }
      res.slice(0, 30).forEach(function (r) {
        var item = document.createElement('div');
        item.className = 'search-item';
        item.innerHTML = '<div class="si-chapter">' + escapeText(r.title) + '</div><div class="si-snippet">' + r.snippet + '</div>';
        item.addEventListener('click', function () {
          go(r.key);
          panel.classList.add('hidden');
        });
        results.appendChild(item);
      });
      panel.classList.remove('hidden');
    }

    input.addEventListener('input', function () {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(doSearch, 150);
    });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') doSearch();
      if (e.key === 'Escape') { panel.classList.add('hidden'); input.blur(); }
    });
    document.addEventListener('click', function (e) {
      if (!panel.contains(e.target) && e.target !== input) panel.classList.add('hidden');
    });
    document.addEventListener('keydown', function (e) {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        input.focus();
        input.select();
      }
    });
    // 首页搜索按钮
    $('#btnSearch').addEventListener('click', function () { input.focus(); input.select(); });
  }

  // ---------- 进度条 / 回到顶部 ----------
  function bindProgress() {
    var bar = $('#progressBar');
    var topBtn = $('#btnTop');
    function update() {
      var doc = document.documentElement;
      var total = doc.scrollHeight - doc.clientHeight;
      var pct = total > 0 ? (doc.scrollTop / total) * 100 : 0;
      bar.style.width = pct + '%';
      topBtn.classList.toggle('hidden', doc.scrollTop < 300);
    }
    window.addEventListener('scroll', update, { passive: true });
    update();
    topBtn.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: 'smooth' }); });
  }

  // ---------- 复制代码 ----------
  function bindCopy() {
    document.addEventListener('click', function (e) {
      var btn = e.target.closest('.code-copy');
      if (!btn) return;
      var pre = btn.closest('pre');
      var code = pre.querySelector('code');
      var text = code.innerText || code.textContent;
      function done() {
        btn.textContent = '已复制 ✓';
        setTimeout(function () { btn.textContent = '复制'; }, 1500);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(); });
      } else { fallback(); }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (err) { btn.textContent = '复制失败'; }
        document.body.removeChild(ta);
      }
    });
  }

  // ---------- 主题 / 字号 ----------
  function bindPrefs() {
    var themeBtn = $('#btnTheme');
    function applyTheme(dark) {
      document.body.classList.toggle('dark', dark);
      themeBtn.textContent = dark ? '☀️' : '🌙';
      try { localStorage.setItem('agent-course-dark', dark ? '1' : '0'); } catch (e) {}
    }
    var saved = null;
    try { saved = localStorage.getItem('agent-course-dark'); } catch (e) {}
    applyTheme(saved === '1' || (saved === null && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches));
    themeBtn.addEventListener('click', function () { applyTheme(!document.body.classList.contains('dark')); });

    $('#btnFontUp').addEventListener('click', function () {
      document.body.classList.remove('font-sm');
      document.body.classList.toggle('font-lg', !document.body.classList.contains('font-lg'));
    });
    $('#btnFontDown').addEventListener('click', function () {
      document.body.classList.remove('font-lg');
      document.body.classList.toggle('font-sm', !document.body.classList.contains('font-sm'));
    });
  }

  // ---------- 侧边栏（移动端） ----------
  function bindSidebar() {
    var sidebar = $('#sidebar');
    var mask = $('#sidebarMask');
    function open() { sidebar.classList.add('open'); mask.classList.add('show'); }
    function close() { sidebar.classList.remove('open'); mask.classList.remove('show'); }
    window.closeSidebar = close;
    $('#btnMenu').addEventListener('click', open);
    mask.addEventListener('click', close);
  }

  // ---------- 启动 ----------
  function init() {
    window.Search.buildIndex();
    renderToc();
    renderHome();
    bindSearch();
    bindProgress();
    bindCopy();
    bindPrefs();
    bindSidebar();

    $('#btnStart').addEventListener('click', function () {
      var ch1 = chapterMeta.filter(function (x) { return x.key.indexOf('ch01') === 0; })[0];
      if (ch1) go(ch1.key);
    });
    $('#btnGuide').addEventListener('click', function () {
      var g = chapterMeta.filter(function (x) { return x.key.indexOf('HANDOVER') === 0; })[0];
      if (g) go(g.key);
    });
    $('#btnGuideTop').addEventListener('click', function () {
      var g = chapterMeta.filter(function (x) { return x.key.indexOf('HANDOVER') === 0; })[0];
      if (g) go(g.key);
    });

    // 品牌标题 → 回首页（根路径 /）
    document.querySelector('.brand').addEventListener('click', function (e) {
      e.preventDefault();
      showHome();
    });

    // 深链：#/ch/<key> 或 #home
    var hash = location.hash;
    if (hash.indexOf('#/ch/') === 0) {
      var key = decodeURIComponent(hash.slice(5));
      if (window.BOOK_DATA[key]) go(key);
      else showHome();
    } else {
      showHome();
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
