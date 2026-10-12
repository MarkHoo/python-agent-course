// 运行时集成测试：用最小 DOM 模拟真实执行 main.js 的 init/go，捕获运行时错误
// 用法: node site/tools/test-runtime.js
const fs = require('fs');
const path = require('path');

const SITE = path.join(__dirname, '..');
global.window = global;

function makeEl(tag) {
  const el = {
    tagName: (tag || 'div').toUpperCase(),
    children: [],
    _innerHTML: '',
    _text: '',
    className: '',
    disabled: false,
    onclick: null,
    _attrs: {},
    style: {},
    _listeners: {},
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      toggle(c) { this._set.has(c) ? this._set.delete(c) : this._set.add(c); },
      contains(c) { return this._set.has(c); },
    },
    setAttribute(k, v) { this._attrs[k] = v; },
    getAttribute(k) { return this._attrs[k]; },
    addEventListener(ev, fn) { (this._listeners[ev] = this._listeners[ev] || []).push(fn); },
    dispatch(ev) { (this._listeners[ev] || []).forEach(fn => fn({ preventDefault() {}, target: el, key: '', ctrlKey: false })); },
    appendChild(c) { this.children.push(c); return c; },
    scrollIntoView() { this._scrolled = true; },
    set innerHTML(v) { this._innerHTML = v; this._text = String(v).replace(/<[^>]+>/g, ''); },
    get innerHTML() { return this._innerHTML; },
    set textContent(v) { this._text = String(v); this._innerHTML = ''; },
    get textContent() { return this._text; },
    set value(v) { this._value = v; },
    get value() { return this._value; },
  };
  return el;
}

const registry = {};
function querySelector(sel) {
  if (sel.startsWith('#')) {
    const id = sel.slice(1);
    if (!registry[id]) registry[id] = makeEl('div');
    return registry[id];
  }
  return makeEl('div');
}
function querySelectorAll() { return []; }

global.document = {
  querySelector,
  querySelectorAll,
  getElementById(id) { return registry[id] || (registry[id] = makeEl('div')); },
  createElement: makeEl,
  createTreeWalker() { return { nextNode() { return null; } }; },
  addEventListener() {},
  readyState: 'complete',
  body: makeEl('body'),
  documentElement: { scrollHeight: 1000, clientHeight: 800, scrollTop: 0 },
};
global.history = { replaceState() {}, pushState() {} };
global.addEventListener = function () {};
global.removeEventListener = function () {};
global.localStorage = { getItem() { return null; }, setItem() {} };
global.location = { hash: '', href: '' };
global.matchMedia = function () { return { matches: false }; };
global.scrollTo = function () {};
global.NodeFilter = { SHOW_TEXT: 4 };

fs.readdirSync(path.join(SITE, 'data')).forEach(f => eval(fs.readFileSync(path.join(SITE, 'data', f), 'utf8')));
eval(fs.readFileSync(path.join(SITE, 'js', 'highlight.js'), 'utf8'));
eval(fs.readFileSync(path.join(SITE, 'js', 'markdown.js'), 'utf8'));
eval(fs.readFileSync(path.join(SITE, 'js', 'search.js'), 'utf8'));
eval(fs.readFileSync(path.join(SITE, 'js', 'main.js'), 'utf8'));

let fail = 0;
function expect(label, cond) { console.log((cond ? '  ✓ ' : '  ✗ ') + label); if (!cond) fail++; }

console.log('=== 运行时集成测试 ===');
try { expect('init 正常执行（无异常）', true); } catch (e) { console.log('  ✗ init 异常:', e.message); fail++; }
try { document.querySelector('#btnStart').dispatch('click'); expect('「开始阅读」跳转第 1 章无异常', true); }
catch (e) { console.log('  ✗ 异常:', e.message); fail++; }
try { document.querySelector('#btnGuide').dispatch('click'); expect('「工程交接说明」跳转无异常', true); }
catch (e) { console.log('  ✗ 异常:', e.message); fail++; }
try { const i = document.querySelector('#searchInput'); i.value = 'LangGraph'; i.dispatch('input'); expect('搜索执行无异常', true); }
catch (e) { console.log('  ✗ 搜索异常:', e.message); fail++; }
try { document.querySelector('#btnTheme').dispatch('click'); document.querySelector('#btnFontUp').dispatch('click'); expect('主题/字号切换无异常', true); }
catch (e) { console.log('  ✗ 异常:', e.message); fail++; }
try { document.querySelector('#btnNext').dispatch('click'); expect('下一章跳转无异常', true); }
catch (e) { console.log('  ✗ 异常:', e.message); fail++; }

console.log('\n' + (fail === 0 ? '✅ 运行时集成测试全部通过（无 ReferenceError）' : '❌ 失败 ' + fail + ' 项'));
process.exit(fail === 0 ? 0 : 1);
