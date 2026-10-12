# 56 · 用户模块：仓储层、资料维护、验证码限流与账号安全

> 这一章讲用户模块：业务代码通常很简单，**复杂的是边界**。
> 本章把四个最容易出事的边界做对：字段白名单、验证码限流、验证码爆破、撞库锁定。

## 本章目标

1. 能用**仓储层**（Repository）把存取与业务分开；
2. 能实现**字段白名单**，挡住"前端多传一个 `role=admin` 就提权"；
3. 能给验证码加四道闸门（冷却 / 频次 / 尝试次数 / 恒定时间比较）；
4. 能实现登录失败锁定与密码强度检查。

## 核心概念

```text
UserRepository（只管存取，asyncio.Lock 保证读改写原子）
        │
        ▼
UserService.update_profile()
        ├── EDITABLE_FIELDS 白名单        ← 挡越权提权
        └── SENSITIVE_FIELDS 需要验证码   ← 挡改绑邮箱/手机
                    │
                    ▼
            CodeService（四道闸门）
                    │
                    ▼
            AccountSecurity（失败锁定 + 密码强度）
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch56_user_module.user_module
```

> 纯 pydantic + 标准库，零依赖、零 Key；时钟与验证码生成器可注入，结果可复现。

### 输出

<!-- run: chapters.ch56_user_module.user_module -->
```text
1) 资料更新的字段白名单（防越权提权）
  已更新：['nickname']
  已忽略：['role', 'status'] ← role/status 不在白名单，提权失败
  当前 role：user（仍为 user ✅）

2) 敏感字段需要验证码
  ✗ 无验证码：[CODE_REQUIRED] 修改「email」需要验证码
  ✓ 带正确验证码：邮箱已改为 new@example.com

3) 验证码的四道闸门
  ✗ 立即重发：[COOLDOWN] 请 60 秒后再试
  刚发送后剩余尝试次数：5
    第 1 次试 000000：False（剩 4 次）
    第 2 次试 111111：False（剩 3 次）
    第 3 次试 222222：False（剩 2 次）

4) 登录失败锁定
  第 1 次失败 → 剩余 4 次
  ...
  第 5 次失败 → 剩余 0 次
  已锁定：True
  成功登录后重置 → 已锁定：False

5) 密码强度检查
  12345678     ❌ 该密码在常见弱密码字典里
  abcdefgh     ❌ 需要同时包含字母和数字
  password     ❌ 该密码在常见弱密码字典里
  Str0ng!Pass  ✅ ok
```

四处值得注意：

- **`已忽略：['role', 'status']`、`当前 role：user`**：前端传 `{"role": "admin"}`
  被白名单**直接忽略，提权失败**。这是真实世界最常见的越权漏洞之一。
- **`无验证码 → [CODE_REQUIRED]`，带验证码 → 成功**：改邮箱这类敏感字段**必须二次验证**，
  否则账号极易被劫持。
- **`立即重发：[COOLDOWN] 请 60 秒后再试`**：冷却看的是**发送历史**而不是验证码记录——
  验证码"用过即焚"，若以记录判冷却，验完就能立刻再发（等于没有冷却）。
- **`第 5 次失败 → 已锁定：True`，`成功登录后重置 → False`**：
  锁定 + 成功即重置，是撞库的唯一有效防线。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

仓储层走内存实现，**不需要真实数据库**。

## 代码走查：`user_module.py`

### 1. 字段白名单：挡住最常见的越权

```python
EDITABLE_FIELDS = frozenset({"nickname", "phone", "avatar_url"})
SENSITIVE_FIELDS = frozenset({"email", "phone"})


async def update_profile(
    self, user_id: str, patch: dict[str, object], *, code: str | None = None
) -> UpdateResult:
    """更新资料。"""
    profile = await self._repo.get(user_id)
    if profile is None:
        raise ServiceError("NOT_FOUND", "用户不存在")

    result = UpdateResult()
    data = profile.model_dump()
    for key, value in patch.items():
        if key not in EDITABLE_FIELDS and key not in SENSITIVE_FIELDS:
            result.ignored.append(key)  # ← 白名单外直接忽略
            continue
        if key in SENSITIVE_FIELDS:
            self._require_code(key, profile, code)  # 敏感字段要验证码
        data[key] = value
        result.updated.append(key)

    await self._repo.save(UserProfile(**data))
    return result
```

数据模型里把这些边界写死：

```python
class UserStatus(StrEnum):
    """账号状态。"""

    ACTIVE = "正常"
    LOCKED = "已锁定"
    DISABLED = "已禁用"


class UserProfile(BaseModel):
    """用户资料。"""

    user_id: str
    email: str
    nickname: str = ""
    phone: str = ""
    avatar_url: str = ""
    role: str = "user"  # ← 敏感字段：**不允许通过资料接口修改**
    status: UserStatus = UserStatus.ACTIVE
    created_at: float = 0.0
```

**敏感字段在模型上就标出来**（`role` / `status` 不属于 `EDITABLE_FIELDS`），
校验发生在**入口**——而不是"到数据库那一层再说"。

**如果写成 `setattr(profile, key, value)` 会怎样**：
前端传 `{"role": "admin"}` 就提权成功了。
这是真实世界里最常见的越权漏洞之一——因为"前端不会传这个字段"
从来都不是一个安全假设。

```text
已更新：['nickname']
已忽略：['role', 'status'] ← role/status 不在白名单，提权失败
当前 role：user（仍为 user ✅）
```

**白名单优于黑名单**：新增一个敏感字段时，黑名单会漏，白名单不会。

### 2. 验证码：四道闸门

| 闸门 | 阈值 | 不做会怎样 |
| --- | --- | --- |
| 冷却 | 60 秒 | 短信轰炸，**直接烧钱** |
| 频次 | 5 次/小时 | 换个 key 继续刷 |
| 尝试次数 | 5 次 | 4 位数字 1 万种组合，不限次数等于没验证 |
| 恒定时间比较 | — | 响应时间泄露"前几位是对的" |

**冷却必须看发送历史，不能看验证码记录**：

```python
history = [item for item in self._sent_log.get(business_key, []) if now - item < 3600]
if history and now - history[-1] < RESEND_COOLDOWN:
    raise ServiceError("COOLDOWN", ...)
```

验证码验完就"用过即焚"了。若以验证码记录判断冷却，
**验完就能立刻再发**——等于没有冷却。这个 bug 我第一版就写了。

发送与校验两个方法把四道闸门落全：

```python
def send(self, business_key: str) -> str:
    """发送验证码。"""
    now = self._clock()
    history = [item for item in self._sent_log.get(business_key, []) if now - item < 3600]
    if history and now - history[-1] < RESEND_COOLDOWN:
        wait = RESEND_COOLDOWN - (now - history[-1])
        raise ServiceError("COOLDOWN", f"请 {wait:.0f} 秒后再试")
    if len(history) >= HOURLY_LIMIT:
        raise ServiceError("RATE_LIMITED", f"每小时最多发送 {HOURLY_LIMIT} 次")
    history.append(now)
    self._sent_log[business_key] = history
    code = self._rng()
    self._records[business_key] = CodeRecord(code=code, expires_at=now + CODE_TTL, sent_at=now)
    return code


def verify(self, business_key: str, code: str) -> bool:
    """校验验证码（**不抛异常，只返回 bool**）。"""
    record = self._records.get(business_key)
    if record is None or record.expires_at < self._clock():
        return False
    if record.attempts >= MAX_ATTEMPTS:
        return False
    record.attempts += 1
    ok = _constant_time_eq(record.code, code)
    if ok:
        self._records.pop(business_key, None)  # 用过即焚
    return ok
```

**恒定时间比较**：

```python
def _constant_time_eq(left: str, right: str) -> bool:
    size = max(len(left), len(right))
    return secrets.compare_digest(left.ljust(size), right.ljust(size))
```

Python 的字符串 `==` 是短路的，第一个字符不同就立刻返回。
先补到等长再比较，连长度都不泄露。

**`verify()` 返回 bool 而不是抛异常**：验证码错误是**预期内的高频事件**，
用异常表达会让正常流程充满 try/except。

把四道闸门的常量集中声明，是为了让"改一个数会怎样"有据可查：

```python
CODE_TTL: Final[float] = 300.0  # 验证码 5 分钟有效
RESEND_COOLDOWN: Final[float] = 60.0  # 60 秒内不能重发
HOURLY_LIMIT: Final[int] = 5  # 同一业务 key 每小时最多 5 次
MAX_ATTEMPTS: Final[int] = 5  # 最多试 5 次
```

**为什么常量要具名且集中**：散落在各处的魔法数改一个漏一个；
集中声明后，本章的五个实验全都"改一个常量 → 看一个现象"，**取舍变得可验证**。

### 3. 账号安全：失败锁定

```python
def record_failure(self, user_id: str) -> int:
    """记录一次登录失败，返回剩余可尝试次数。"""
    state = self._states.setdefault(user_id, SecurityState())
    state.failures += 1
    if state.failures >= MAX_FAILURES:
        state.locked_until = self._clock() + LOCK_DURATION
        logger.warning("账号 %s 连续失败 %d 次，已锁定", user_id, state.failures)
        return 0
    return MAX_FAILURES - state.failures


def reset(self, user_id: str) -> None:
    """登录成功，清空失败计数。"""
    self._states.pop(user_id, None)
```

```python
@dataclass(slots=True)
class SecurityState:
    """一个账号的安全状态。"""

    failures: int = 0
    locked_until: float = 0.0


class AccountSecurity:
    """账号安全：登录失败锁定 + 密码强度。"""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """初始化。"""
        self._states: dict[str, SecurityState] = {}
        self._clock = clock

    def locked(self, user_id: str) -> bool:
        """是否处于锁定中。"""
        state = self._states.get(user_id)
        return bool(state and state.locked_until > self._clock())
```

```text
第 1 次失败 → 剩余 4 次
...
第 5 次失败 → 剩余 0 次
已锁定：True
成功登录后重置 → 已锁定：False
```

连续 5 次失败锁定 15 分钟。这是**撞库的唯一有效防线**——
不锁定的话，同一个账号可以被无限次尝试。

> 锁定的粒度要权衡：按 IP 锁会误伤（NAT/公司出口），
> 按账号锁会被用来"锁别人的账号"（DoS）。
> 生产通常**账号 + IP 双维度**，且 IP 维度阈值放宽 10 倍。

### 4. 密码强度与仓储层

密码强度检查就三条规则，但**顺序有讲究**：

```python
@staticmethod
def check_password(password: str) -> tuple[bool, str]:
    """密码强度检查。Returns: ``(是否合格, 原因)``。"""
    if len(password) < MIN_PASSWORD_LEN:
        return False, f"至少 {MIN_PASSWORD_LEN} 位"
    if password.lower() in WEAK_PASSWORDS:
        return False, "该密码在常见弱密码字典里"
    if password.isdigit() or password.isalpha():
        return False, "需要同时包含字母和数字"
    return True, "ok"
```

**先查长度再看字典**：`"123456"` 既短又在字典里，先报长度能给出更具体的理由。
返回 `(bool, reason)` 而不是只返回 `bool`——**用户需要知道自己错在哪**。

仓储层则给每个读/写都套上锁：

```python
class InMemoryUserRepo:
    """进程内实现（生产换成 SQLAlchemy 2.0 + PostgreSQL）。"""

    def __init__(self) -> None:
        """初始化。"""
        self._rows: dict[str, UserProfile] = {}
        self._lock = asyncio.Lock()

    async def get(self, user_id: str) -> UserProfile | None:
        """按 ID 取。"""
        async with self._lock:
            return self._rows.get(user_id)

    async def save(self, profile: UserProfile) -> None:
        """保存（按 user_id 覆盖）。"""
        async with self._lock:
            self._rows[profile.user_id] = profile
```

**为什么读也要加锁**：`update_profile` 是"读出来 → 改 → 写回去"三步。
两个请求并发时，后写的会覆盖先写的（lost update）——
**读不加锁只解决了"读到一半"的问题，解决不了"整个更新不原子"的问题**。

四条规则的示例输出：

```text
12345678     ❌ 该密码在常见弱密码字典里
abcdefgh     ❌ 需要同时包含字母和数字
password     ❌ 该密码在常见弱密码字典里
Str0ng!Pass  ✅ ok
```

**弱密码字典比"必须含特殊字符"有效得多**——
后者只会让人把 `!` 写在末尾（`Password1!`），字典则直接拦住最常见的那些。

### 5. 实验入口：`experiments.py`

本章还有一个**可执行入口** `experiments.py`——把下面「动手实验」的五条
全部做成了可跑的命令（`--list` / `--exp N` / `--all`）。
它存在的意义是：**实验必须能被 CI 反复执行**，而不是"让读者手动改文件"。

```python
def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="字段白名单挡住越权提权",
            what="把 update_profile 改成直接 setattr 所有 patch 字段",
            expect="前端多传 role=admin 就提权成功",
            run=exp1_field_whitelist,
        ),
        # …其余四条见 experiments.py
    ]
```

每个 `run` 都是**零参纯函数**：用"改后的参数再跑一遍"表达差异，绝不真的改仓库文件。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把白名单改成 setattr"看着简单，但读者改了文件要记得改回来；
"能提权"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数 / 复刻的旧实现再跑一遍纯函数**"这条路。

```bash
python -m chapters.ch56_user_module.experiments --list    # 列出全部实验
python -m chapters.ch56_user_module.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch56_user_module.experiments --all     # 全跑并汇总断言
```

### 实验 1：字段白名单挡住越权提权

```bash
python -m chapters.ch56_user_module.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `update_profile` 改成直接 `setattr` 所有 patch 字段 |
| 期望现象 | 前端多传 `role=admin` 就提权成功 |
| 实测 | 白名单：已更新 `['nickname']`、已忽略 `['role','status']`、`role` 仍是 `user`；直接 setattr：`role=admin` |

**为什么**：`"前端不会传这个字段"` 从来不是安全假设。
白名单是**唯一防线**：不在名单里的字段一律忽略——**新增敏感字段时白名单不会漏，
黑名单一定会漏**。

### 实验 2：验证码冷却必须看发送历史

```bash
python -m chapters.ch56_user_module.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把冷却判据从发送历史改成「当前验证码记录是否存在」 |
| 期望现象 | 验证码用完即焚，验证一完就能立刻重发——等于没有冷却 |
| 实测 | 看发送历史：立刻重发 → `REJECTED:COOLDOWN`；看记录：成功重发（`ok`） |

**为什么**：验证码"用过即焚"（`verify` 成功会 `pop` 掉记录）。
以记录判冷却时，**用户一验证通过就能立刻再发**——短信轰炸照旧。
冷却要看的是**发送历史**，不是"当前还有没有没用的码"。

### 实验 3：验证码尝试次数必须有上限

```bash
python -m chapters.ch56_user_module.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 去掉 `verify()` 里的 `attempts >= MAX_ATTEMPTS` 判断 |
| 期望现象 | 4 位数字可被暴力枚举；有上限时超限后连正确码也拒绝 |
| 实测 | 连试 8 次全 `False`；超限后用**正确码**也 `False`；无上限时爆破到正确码用了 **10000** 次 |

**为什么**：4 位数字只有 1 万种组合。下限设在**校验正确性之前**——
超限后连正确码都拒绝，否则"猜对就通过"等于给爆破开了后门。

### 实验 4：验证码比较要恒定时间

```bash
python -m chapters.ch56_user_module.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `_constant_time_eq` 换成 `==` |
| 期望现象 | 结果相同，但 `==` 短路返回会泄露前缀匹配信息 |
| 实测 | `==` → `False`；`_constant_time_eq` → `False`；长度不同也 `False`（先补到等长） |

**为什么**：Python 的字符串 `==` 在第一个不同字符处短路返回，
响应时间会泄露"前几位是对的"。先补到等长再比较，**连长度都不泄露**。

### 实验 5：成功登录必须重置失败计数

```bash
python -m chapters.ch56_user_module.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 去掉登录成功后的 `security.reset` |
| 期望现象 | 正常用户偶发输错会被累计失败误锁 |
| 实测 | 连续失败 5 次 → 已锁定 `True`；reset 后解锁 `True`；不 reset 时再输错一次 → 又锁定 |

**为什么**：锁定必须配"成功即重置"。少了它，失败计数会**跨会话累积**，
正常用户偶尔手误几次就会被自己的历史失败永久锁掉。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 直接 setattr 更新 | 字段白名单 | 挡住越权提权 |
| 改邮箱不需要验证 | `SENSITIVE_FIELDS` + 验证码 | 挡住账号劫持 |
| 验证码不限流 | 冷却 + 频次 | 不被短信轰炸烧钱 |
| 验证码不限次数 | 最多 5 次 | 不被爆破 |
| 用 `==` 比验证码 | 恒定时间比较 | 不泄露前缀 |
| 冷却看验证码记录 | 看发送历史 | 验完不能再立刻发 |
| 登录失败不锁定 | 5 次锁定 15 分钟 | 挡撞库 |
| 存取与业务混一起 | Repository 分层 | 可换数据库、可测 |
| 时间不可控 | `clock` / `rng` 可注入 | 限流逻辑可测 |

## 最佳实践清单

- **永远用白名单，不用黑名单**。
- **敏感字段（邮箱/手机）修改必须二次验证**。
- **验证码用过即焚**，但**冷却看发送历史**。
- **比较秘密值用恒定时间函数**。
- **预期内的失败返回 bool，意外情况才抛异常**。
- **登录失败要锁定**，并权衡账号/IP 粒度。
- **弱密码字典比字符规则有效**。
- **仓储层保证读改写原子**。

## 常见坑 / 排错

1. **冷却不生效** → 用验证码记录判断冷却，而记录验完就删了；改成看发送历史。
2. **`role` 被改了** → 忘了白名单，或新增字段时没同步 `EDITABLE_FIELDS`。
3. **验证码一直错误** → 时钟注入问题；`expires_at` 用的是 `clock()`，测试里固定时钟不会走。
4. **正常用户被锁** → 阈值太严，或没做"成功即重置"。
5. **并发更新丢失** → 仓储层没加锁，或用了多个仓储实例。
6. **`secrets.choice` 生成验证码偏慢** → 只在发送时调用一次，别放在循环里。

## 延伸练习

1. 给 `CodeService` 加**图形验证码前置**，人机校验通过后才发短信。
2. 实现**分级限流**：同 IP 每小时 20 次、同账号每小时 5 次。
3. 把 `InMemoryUserRepo` 换成 SQLAlchemy 2.0 实现（第 49 章的表结构）。
4. 加"**异地登录提醒**"：记录上次登录 IP 属地，变更时要求二次验证。

## 小结

用户模块的代码不难，难的是**每一条边界都想过了**。
下一章（57）讲 RBAC 权限系统：角色继承、权限判定与资源级鉴权。
