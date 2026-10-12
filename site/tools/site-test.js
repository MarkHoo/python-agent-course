// 站点逻辑真实验证（Node）：渲染全部页面 + 搜索 + 高亮 + 列表 + 锚点一致性
// 用法: node site/tools/site-test.js
const fs = require('fs');
const path = require('path');

const SITE = path.join(__dirname, '..');
global.window = global;

const dataDir = path.join(SITE, 'data');
const dataFiles = fs.readdirSync(dataDir).filter(f => f.endsWith('.js'));
dataFiles.forEach(f => eval(fs.readFileSync(path.join(dataDir, f), 'utf8')));
const count = Object.keys(global.BOOK_DATA || {}).length;
console.log('数据页数:', count);

eval(fs.readFileSync(path.join(SITE, 'js', 'highlight.js'), 'utf8'));
eval(fs.readFileSync(path.join(SITE, 'js', 'markdown.js'), 'utf8'));
eval(fs.readFileSync(path.join(SITE, 'js', 'search.js'), 'utf8'));

let fail = 0;

// 1) 渲染全部页面
console.log('\n=== 渲染测试 ===');
for (const k of Object.keys(global.BOOK_DATA)) {
  try {
    const r = global.Markdown.render(global.BOOK_DATA[k]);
    const pre = (r.html.match(/<pre>/g) || []).length;
    const preC = (r.html.match(/<\/pre>/g) || []).length;
    const tb = (r.html.match(/<table>/g) || []).length;
    const tbC = (r.html.match(/<\/table>/g) || []).length;
    if (!r.html.trim()) throw new Error('渲染为空');
    if (pre !== preC) throw new Error('pre 不配对: ' + pre + '/' + preC);
    if (tb !== tbC) throw new Error('table 不配对: ' + tb + '/' + tbC);
  } catch (e) {
    fail++;
    console.log('  ✗', k, '→', e.message);
  }
}
console.log('  ✓ 渲染 ' + (count - fail) + '/' + count + ' 页');

// 2) 搜索
console.log('\n=== 搜索测试 ===');
global.Search.buildIndex();
const cases = [
  ['Tool', 'ch04'], ['MCP', 'ch06'], ['RAG', 'ch08'], ['Milvus', 'ch11'],
  ['LangGraph', 'ch25'], ['Redis', 'ch35'], ['Transformer', 'ch42'],
  ['简历', 'ch44'], ['RBAC', 'ch57'], ['SSE', 'ch20'],
];
for (const [q, chPrefix] of cases) {
  const res = global.Search.search(q);
  const hit = res.some(r => r.key.startsWith(chPrefix));
  console.log('  搜索「' + q + '」→', res.length, '条' + (hit ? ' ✓' : ' ✗ 缺 ' + chPrefix));
  if (!res.length || !hit) fail++;
}
const multi = global.Search.search('RAG 检索');
console.log('  搜索「RAG 检索」→', multi.length, '条（AND 语义）');
if (!multi.length) fail++;

// 3) 高亮
console.log('\n=== 高亮测试 ===');
const hl = global.Highlighter.highlight('def foo(x: int) -> str:\n    return str(x)  # 注释', 'python');
if (!hl.includes('<span')) { console.log('  ✗ 未产生高亮 span'); fail++; }
else console.log('  ✓ Python 关键字/注释高亮正常');
const hl2 = global.Highlighter.highlight('{"a": 1, "b": "x"}', 'json');
if (!hl2.includes('<span')) { console.log('  ✗ JSON 高亮失败'); fail++; }
else console.log('  ✓ JSON 字符串/数字高亮正常');

// 4) 列表 / 表格 / 引用
console.log('\n=== 结构渲染测试 ===');
const nestedMd = '- 一级 A\n  - 二级 A1\n  - 二级 A2\n- 一级 B\n1. 有序一\n2. 有序二';
const nl = global.Markdown.render(nestedMd).html;
function countTag(html, tag) { return (html.match(new RegExp('<' + tag + '( |>|$)', 'g')) || []).length; }
const ul = countTag(nl, 'ul'), ulc = (nl.match(/<\/ul>/g) || []).length;
const ol = countTag(nl, 'ol'), olc = (nl.match(/<\/ol>/g) || []).length;
const li = countTag(nl, 'li'), lic = (nl.match(/<\/li>/g) || []).length;
if (ul === ulc && ol === olc && li === lic && ul === 2 && ol === 1 && li === 6) console.log('  ✓ 嵌套列表结构正确');
else { console.log('  ✗ 列表结构异常', { ul, ulc, ol, olc, li, lic }); fail++; }

// 5) 锚点一致性（h2 标题 → id）
console.log('\n=== 锚点一致性测试 ===');
function slugifyJS(text) {
  return 'sec-' + text.replace(/[^\w\u4e00-\u9fa5]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60);
}
let anchorFail = 0;
for (const k of Object.keys(global.BOOK_DATA)) {
  const md = global.BOOK_DATA[k];
  const rendered = global.Markdown.render(md);
  const noCode = md.replace(/^```[\s\S]*?^```/gm, '');
  const re = /^##\s+(.+)$/gm;
  let m;
  while ((m = re.exec(noCode)) !== null) {
    const title = m[1].replace(/\s*★+.*$/, '').trim();
    if (!rendered.html.includes('id="' + slugifyJS(title) + '"')) {
      if (anchorFail < 8) console.log('  ✗ ' + k + ' 缺锚点: ' + title);
      anchorFail++;
    }
  }
}
console.log('  锚点检查: ' + (anchorFail === 0 ? '全部 h2 锚点一致 ✓' : anchorFail + ' 处不一致 ✗'));
if (anchorFail) fail += anchorFail;

// 6) 首页分组覆盖完整性（main.js 的 groups 必须覆盖全部数据 key）
console.log('\n=== 首页分组覆盖测试 ===');
try {
  const mainSrc = fs.readFileSync(path.join(SITE, 'js', 'main.js'), 'utf8');
  const gm = mainSrc.match(/var groups = \[([\s\S]*?)\n    \];/);
  if (!gm) throw new Error('未找到 groups 定义');
  const groups = eval('[' + gm[1] + ']');
  const keys = Object.keys(global.BOOK_DATA);
  const covered = new Set();
  for (const g of groups) for (const k of keys) if (g.keys.some(p => k.indexOf(p) === 0)) covered.add(k);
  const uncovered = keys.filter(k => !covered.has(k));
  const empty = groups.filter(g => keys.filter(k => g.keys.some(p => k.indexOf(p) === 0)).length === 0);
  if (uncovered.length) { console.log('  ✗ 未被分组覆盖:', uncovered.join(', ')); fail += uncovered.length; }
  if (empty.length) { console.log('  ✗ 空分组:', empty.map(g => g.label).join(', ')); fail += empty.length; }
  if (!uncovered.length && !empty.length) console.log('  ✓ ' + keys.length + ' 个条目全部被 ' + groups.length + ' 个分组覆盖');
} catch (e) { console.log('  ✗ 分组检查异常:', e.message); fail++; }

console.log('\n' + (fail === 0 ? '✅ 全部测试通过' : '❌ 失败项: ' + fail));
process.exit(fail === 0 ? 0 : 1);
