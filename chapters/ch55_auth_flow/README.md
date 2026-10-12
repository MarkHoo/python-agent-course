# 55 · 用户鉴权流程：密码哈希、JWT 签发与刷新令牌轮换

> 这一章讲用户鉴权：看起来是「调个库」。
> 但**最容易错的三件事**都在这里：密码怎么存、token 怎么签、refresh 怎么换。
> 本章全部用标准库实现（不引第三方 JWT 库），把每一步摊开。

## 本章目标

1. 知道为什么密码必须用 `scrypt`/`argon2` 而不是 `sha256`；
2. 能手写一个 HS256 JWT，并理解 `exp` / `type` / `jti` 各自防什么；
3. 理解为什么 refresh token **必须轮换**、以及**重放检测**为什么能救命；
4. 知道登录失败时为什么不区分"用户不存在"与"密码错误"。

## 核心概念

```text
注册 → hash_password()  scrypt + 随机盐 → "salt$key"
登录 → verify_password() 恒定时间比较 → 签发 access(15m) + refresh(7d)
刷新 → consume() 标记旧的 used，签发新的（同 family）
       └─ 旧的再用一次？→ REPLAY_DETECTED → 吊销整个 family
登出 → revoke_family() 所有设备一起下线
```

## 运行

```bash
cd python-agent-course
uv run python -m chapters.ch55_auth_flow.auth_flow
```

> 纯标准库实现（`hashlib` + `hmac` + `secrets`），零依赖、零 Key；时间与密钥可注入。

### 输出

<!-- run: chapters.ch55_auth_flow.auth_flow -->
```text
1) 密码哈希
  存储串：77778fce72543467965b4545…（salt + scrypt 派生值，共 97 字符）
  正确密码：True
  错误密码：False
  两次哈希同一密码结果不同：True（随机盐）

2) 登录与访问
  access：eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ…
  refresh：eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ…
  whoami：u1

3) 刷新轮换（旧的立即失效，签发新的）
  新的 refresh 与旧的不同：True
  旧 refresh 再用一次：[REPLAY_DETECTED] 检测到令牌重放，已吊销该用户全部会话

4) 重放检测的后果
  该令牌族已吊销：True
  即便用**没用过的新 token** 也失效：[FAMILY_REVOKED] 该令牌族已被吊销，请重新登录

5) 令牌校验的三类拒绝
  ✗ refresh 当 access 用：[WRONG_TYPE] 期望 access 令牌，实际是 refresh
  ✗ 已过期：[EXPIRED] 令牌已过期
  ✗ 密钥不对：[BAD_SIGNATURE] 签名校验失败

要点：refresh token 泄露不是『换个新的就行』——
  必须**轮换 + 重放检测**，否则攻击者可以无限期续期。
```

四处值得注意：

- **`两次哈希同一密码结果不同：True`**：随机盐的效果。**相同密码不会得到相同哈希**，
  彩虹表失效。
- **`旧的 refresh 再用一次 → [REPLAY_DETECTED]`**：轮换的本质是"旧的立即作废"。
  它一旦被再次使用，说明**要么是攻击者偷到了，要么是前端并发刷新**——
  系统一律判为泄露。
- **`即便用没用过的新 token 也失效：[FAMILY_REVOKED]`**：**重放检测吊销的是整个"族"**，
  不是那一张。攻击者手里的新 token 也一起作废。
- **三类拒绝各带错误码**：`WRONG_TYPE` / `EXPIRED` / `BAD_SIGNATURE`，
  错误文案精确到"期望什么、实际什么"。

## 环境准备

```bash
# 项目根目录，一次装齐核心依赖（langchain-core / pydantic / rich 等）
uv venv .venv --python 3.12        # Windows: py -3.12 -m venv .venv
uv pip install -e .
cp .env.example .env               # 需要真实模型时填 OPENAI_API_KEY / OPENAI_BASE_URL
```

> **没有 API Key 也能跑**：`build_chat_model()` 在缺 Key 或 `OFFLINE=1` 时自动降级为离线
> `EchoChatModel`，链路完整走一遍。测试与 CI 一律强制 `OFFLINE=1`。

密码哈希用标准库 `hashlib`/`hmac`，JWT 用自研实现（不依赖 pyjwt），**不需要额外依赖或 Key**。

## 代码走查：`auth_flow.py`

### 1. 密码哈希：为什么不能用 sha256

```python
derived = hashlib.scrypt(password.encode(), salt=raw_salt, n=2**14, r=8, p=1, dklen=32)
```

**md5 / sha1 / 裸 sha256 是为速度设计的**——GPU 上一秒能试几十亿个。
`scrypt` 是**内存硬性**（memory-hard），专门为抗 ASIC/GPU 暴力破解而生。

三个要素缺一不可：

| 要素 | 实现 | 不做会怎样 |
| --- | --- | --- |
| 慢哈希 | `scrypt` | 撞库一秒几亿次 |
| 随机盐 | `secrets.token_bytes(16)` | 彩虹表直接命中；相同密码哈希相同 |
| 恒定时间比较 | `hmac.compare_digest` | 通过响应时间逐字节爆破 |

```python
def hash_password(password: str, *, salt: bytes | None = None) -> str:
    """哈希密码：``scrypt`` + 随机盐。"""
    raw_salt = salt if salt is not None else secrets.token_bytes(SALT_LEN)
    derived = hashlib.scrypt(
        password.encode("utf-8"), salt=raw_salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=KEY_LEN
    )
    return f"{raw_salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码。**恒定时间比较**，避免通过响应时间逐字节爆破。"""
    try:
        salt_hex, _key_hex = stored.split("$", maxsplit=1)
    except ValueError:
        return False
    candidate = hash_password(password, salt=bytes.fromhex(salt_hex))
    return hmac.compare_digest(candidate, stored)
```

```text
两次哈希同一密码结果不同：True（随机盐）
```

### 2. 自签 JWT：四个必填字段

```python
def sign_token(payload, *, secret, ttl, clock) -> str:
    body = {**payload, "iat": int(now), "exp": int(now + ttl)}
```

| 字段 | 作用 |
| --- | --- |
| `sub` | 用户标识 |
| `type` | **防止 refresh 当 access 用** |
| `exp` | 过期时间 |
| `jti` | 令牌 ID，**吊销与重放检测靠它** |

```python
def sign_token(
    payload: dict[str, object], *, secret: str, ttl: float, clock: Callable[[], float] = time.time
) -> str:
    """签发一个 JWT（HS256）。"""
    now = clock()
    body = {**payload, "iat": int(now), "exp": int(now + ttl)}
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body_encoded = _b64url_encode(json.dumps(body, separators=(",", ":")).encode())
    signing_input = f"{header}.{body_encoded}".encode("ascii")
    signature = _b64url_encode(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest())
    return f"{header}.{body_encoded}.{signature}"
```

校验顺序**不能变**：

```python
def verify_token(
    token: str, *, secret: str, expected: TokenType, clock: Callable[[], float] = time.time
) -> dict[str, object]:
    """校验 JWT：签名 → 过期 → 类型。"""
    parts = token.split(".")
    if len(parts) != 3:
        raise TokenError("MALFORMED", "令牌格式不正确")
    header, body_encoded, signature = parts
    expected_signature = _b64url_encode(
        hmac.new(secret.encode(), f"{header}.{body_encoded}".encode("ascii"), hashlib.sha256).digest()
    )
    if not hmac.compare_digest(signature, expected_signature):  # ① 先验签名
        raise TokenError("BAD_SIGNATURE", "签名校验失败")
    payload = json.loads(_b64url_decode(body_encoded))
    if payload.get("exp", 0) < clock():  # ② 再验过期
        raise TokenError("EXPIRED", "令牌已过期")
    if payload.get("type") != expected.value:  # ③ 最后验类型
        raise TokenError("WRONG_TYPE", f"期望 {expected.value} 令牌，实际是 {payload.get('type')}")
    return payload
```

**先验签名再验其他**——否则攻击者可以用伪造的 payload 触发各种解析逻辑。

```text
✗ refresh 当 access 用：[WRONG_TYPE] 期望 access 令牌，实际是 refresh
✗ 已过期：[EXPIRED] 令牌已过期
✗ 密钥不对：[BAD_SIGNATURE] 签名校验失败
```

### 3. 刷新轮换 + 重放检测（本章重点）

**问题**：只做"refresh 换 access"的系统，refresh token 一旦泄露，
攻击者可以**无限期**续期。

**解法**：

1. 每次刷新签发一张**新** refresh（新 `jti`），旧的立即标记 `used`；
2. 若一张**已 used** 的 token 又出现 → 判定泄露 → **吊销整个 family**。

```python
if record.used:
    self.revoked_families.add(record.family)
    raise TokenError("REPLAY_DETECTED", "检测到令牌重放，已吊销该用户全部会话")
```

```text
3) 刷新轮换（旧的立即失效，签发新的）
  新的 refresh 与旧的不同：True
  旧 refresh 再用一次：[REPLAY_DETECTED] 检测到令牌重放，已吊销该用户全部会话

4) 重放检测的后果
  该令牌族已吊销：True
  即便用**没用过的新 token** 也失效：[FAMILY_REVOKED] 该令牌族已被吊销
```

**为什么"族"这个粒度**：正常用户和攻击者可能各持有一张有效 token
（攻击者偷到后，用户又刷新了一次）。只吊销被重放的那一张没用，
必须把**同一次登录衍生出来的所有 token** 全部作废，
逼真实用户重新登录——这是唯一能止损的做法。

代价是用户会被踢下线，但这个代价是**必须付的**。

### 4. 刷新存储：`consume()` 是重放检测的落点

"轮换 + 重放检测"听着玄，落地其实就是一个 `consume()`：

```python
@dataclass(frozen=True, slots=True)
class RefreshRecord:
    """一张 refresh token 的记录。"""

    jti: str
    family: str
    user_id: str
    expires_at: float
    used: bool = False


class RefreshStore:
    """refresh token 存储。"""

    records: dict[str, RefreshRecord] = field(default_factory=dict)
    revoked_families: set[str] = field(default_factory=set)

    def issue(self, *, user_id: str, family: str, now: float) -> RefreshRecord:
        """签发一张新 token 记录。"""
        record = RefreshRecord(
            jti=secrets.token_hex(8), family=family, user_id=user_id, expires_at=now + REFRESH_TTL
        )
        self.records[record.jti] = record
        return record

    def consume(self, jti: str, *, now: float) -> RefreshRecord:
        """消费一张 refresh token（**标记已用**）。"""
        record = self.records.get(jti)
        if record is None:
            raise TokenError("UNKNOWN", "refresh token 不存在")
        if record.family in self.revoked_families:
            raise TokenError("FAMILY_REVOKED", "该令牌族已被吊销，请重新登录")
        if record.expires_at < now:
            raise TokenError("EXPIRED", "refresh token 已过期")
        if record.used:
            # ★ 重放检测：用过的令牌又来了 = 泄露
            self.revoked_families.add(record.family)
            raise TokenError("REPLAY_DETECTED", "检测到令牌重放，已吊销该用户全部会话")
        self.records[jti] = RefreshRecord(
            jti=record.jti,
            family=record.family,
            user_id=record.user_id,
            expires_at=record.expires_at,
            used=True,
        )
        return record
```

**四条检查的顺序是有讲究的**：族吊销（`FAMILY_REVOKED`）要在"是否用过"之前判，
否则一张已被吊销族里的旧 token 会先命中 `used` 分支、重复触发吊销——
**同一个错误被报两次，日志里会看不出是哪次泄露触发的**。
`RefreshRecord` 用 `frozen`，`used` 是靠**换一条新记录**表达的，
而不是原地改字段——这样"消费"这个动作天然是原子的。

```python
if user is None or not verify_password(password, user.password_hash):
    raise AuthError("BAD_CREDENTIALS", "邮箱或密码错误")
```

**不区分"用户不存在"与"密码错误"**——否则攻击者可以用响应时间或错误文案
**枚举出哪些邮箱注册过**（账号枚举攻击）。

### 5. 鉴权服务：`refresh()` 把轮换串起来

`AuthService` 把"验签 → 消费 → 轮换"三步串成一次刷新：

```python
def refresh(self, refresh_token: str) -> TokenPair:
    """刷新：**轮换 + 重放检测**。"""
    payload = self._verify_refresh(refresh_token)
    jti = str(payload.get("jti", ""))
    record = self._refresh.consume(jti, now=self.now)
    # 轮换：旧记录已标记 used，签发一张新的（同一 family）
    new_record = self._refresh.issue(user_id=record.user_id, family=record.family, now=self.now)
    return self._pair(record.user_id, new_record)


def _pair(self, user_id: str, record: RefreshRecord) -> TokenPair:
    """签发一对令牌。"""
    access = sign_token(
        {"sub": user_id, "type": TokenType.ACCESS.value, "jti": secrets.token_hex(8)},
        secret=self._secret,
        ttl=ACCESS_TTL,
        clock=self._clock,
    )
    refresh = sign_token(
        {"sub": user_id, "type": TokenType.REFRESH.value, "jti": record.jti, "fam": record.family},
        secret=self._secret,
        ttl=REFRESH_TTL,
        clock=self._clock,
    )
    return TokenPair(access=access, refresh=refresh, family=record.family)
```

**注意 `_pair` 里的两个 `ttl`**：`access` 15 分钟、`refresh` 7 天。
短 access 让"令牌泄露后的可利用窗口"尽量小，长 refresh 让用户不必频繁重登——
**这是安全与体验之间最平衡的一组默认值**（想更快止损就把 access 再缩短，
配合刷新即可）。

### 6. 登录失败不区分原因

本章还有一个**可执行入口** `experiments.py`——把下面「动手实验」的五条
全部做成了可跑的命令（`--list` / `--exp N` / `--all`）。
它存在的意义是：**实验必须能被 CI 反复执行**，而不是"让读者手动改文件"。

```python
def build_experiments() -> list[Experiment]:
    """返回本章全部实验。"""
    return [
        Experiment(
            name="密码必须 scrypt + 盐，不能用 sha256",
            what="把 hash_password 换成裸 sha256（无盐）",
            expect="同一密码两次哈希结果相同、长度骤降，且为速度设计抗不住 GPU 爆破",
            run=exp1_password_hash,
        ),
        # …其余四条见 experiments.py
    ]
```

每个 `run` 都是**零参纯函数**：用"改后的参数再跑一遍"表达差异，绝不真的改仓库文件。

### 7. 时间与密钥可注入

```python
def __init__(self, *, secret: str, clock: Callable[[], float] = time.time) -> None:
```

`clock` 可注入才能测试"过期""轮换窗口"这类逻辑——
否则测试只能 `sleep` 15 分钟。

> 演示里的 `secret="demo-secret-key-change-me"` 加了 `# noqa: S106`：
> 这是在提醒"这只是演示"。**生产的密钥必须走配置/密钥管理**，绝不进代码。

## 动手实验

**这一节的实验全部来自 `experiments.py`，一条命令一件事，不需要改仓库文件。**
"把 scrypt 换成 sha256"看着简单，但读者改了文件要记得改回来；
"会被爆破"没有基准值，跑出什么都算对；文档里的实验不会被 CI 执行。
本章实验统一走"**用改后的参数 / 复刻的旧实现再跑一遍纯函数**"这条路。

```bash
python -m chapters.ch55_auth_flow.experiments --list    # 列出全部实验
python -m chapters.ch55_auth_flow.experiments --exp 1   # 只跑第 1 个
python -m chapters.ch55_auth_flow.experiments --all     # 全跑并汇总断言
```

### 实验 1：密码必须 scrypt + 盐，不能用 sha256

```bash
python -m chapters.ch55_auth_flow.experiments --exp 1
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `hash_password` 换成裸 `sha256`（无盐） |
| 期望现象 | 同一密码两次哈希结果相同、长度骤降，且为速度设计抗不住 GPU 爆破 |
| 实测 | scrypt：两次结果不同（随机盐），存储串 97 字符；sha256：两次结果相同，64 hex |

**为什么**：`md5`/`sha1`/裸 `sha256` 是**为速度设计**的，GPU 上一秒能试几十亿次。
`scrypt` 是**内存硬性**的，专门为抗 ASIC/GPU 而生；随机盐让彩虹表失效。
代价是每次登录多几毫秒——**这几毫秒就是撞库成本的数量级差**。

### 实验 2：比较必须恒定时间

```bash
python -m chapters.ch55_auth_flow.experiments --exp 2
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把 `hmac.compare_digest` 换成 `==` |
| 期望现象 | 结果相同，但 `==` 短路返回，响应时间会泄露前缀匹配信息 |
| 实测 | `==` 比较错误签名 → `False`；`compare_digest` → `False`（**结果一致**） |

**为什么**：两者**功能上等价**，差别在**时间**。`==` 在第一个不同字符处就返回，
攻击者能靠测量响应时间逐字节猜出签名。**安全相关的比较必须恒定时间**——
这种 bug 单看功能测试永远发现不了。

### 实验 3：先验签名，再验过期与类型

```bash
python -m chapters.ch55_auth_flow.experiments --exp 3
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 把校验顺序改成「先验过期、后验签名」 |
| 期望现象 | 同一张伪造令牌，先验过期会走到 payload 解析（错误码 `EXPIRED` 而不是 `BAD_SIGNATURE`） |
| 实测 | 本章顺序 → `BAD_SIGNATURE`；反序 → `expired` |

**为什么**：换成"先验过期"，一张伪造的 payload 会**先进解析逻辑**再被拒绝。
本章先验签名，**伪造的 payload 根本没有机会进入后续处理**——
校验顺序本身就是一道安全边界。

### 实验 4：重放要吊销整个令牌族

```bash
python -m chapters.ch55_auth_flow.experiments --exp 4
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 检测到重放后只吊销被重放的那一张 token |
| 期望现象 | 同一 family 里未被使用的新 token 仍能续期，防不住攻击者 |
| 实测 | 重放旧 token → `REPLAY_DETECTED`；该族已吊销 `True`；再用全新 refresh → `FAMILY_REVOKED` |

**为什么**：攻击者偷到 token 后，用户可能又刷新了一次——于是**正常用户和攻击者
各持有一张有效 token**。只吊销被重放的那一张没用，必须作废**同一次登录衍生出的
全部 token**。代价是把用户也踢下线，但这个代价值得付。

### 实验 5：登录失败不区分账号是否存在

```bash
python -m chapters.ch55_auth_flow.experiments --exp 5
```

| 项 | 值 |
| --- | --- |
| 改了什么 | 让「用户不存在」与「密码错误」返回不同错误码 |
| 期望现象 | 可枚举已注册邮箱；本章两者统一为 `BAD_CREDENTIALS` |
| 实测 | 邮箱不存在 → `BAD_CREDENTIALS`；密码错误 → `BAD_CREDENTIALS`（相同） |

**为什么**：区分两者的话，攻击者用一批邮箱跑一遍登录，
就能**枚举出哪些邮箱注册过**——这是后续钓鱼、撞库的第一步。
统一错误码是零成本的防御。

## 本章的工程化取舍

| 常见做法 | 本章做法 | 收益 |
| --- | --- | --- |
| 用库不解释 | 手写 HS256 JWT | 知道每一段是什么 |
| 只做 refresh 换 access | 轮换 + 重放检测 | 泄露能止损 |
| 令牌不区分类型 | `TokenType` | refresh 不能当 access |
| 没有 jti / family | `jti` + `family` | 可吊销、可检测重放 |
| 密码用快哈希 | `scrypt` + 随机盐 | 抗撞库 |
| 用 `==` 比密码 | `hmac.compare_digest` | 抗时序攻击 |
| 错误提示区分原因 | 统一 `BAD_CREDENTIALS` | 防账号枚举 |
| 时间不可控 | `clock` 可注入 | 时间逻辑可测 |
| 登出只删当前 token | `revoke_family()` | 全设备下线 |

## 最佳实践清单

- **密码用慢哈希 + 随机盐 + 恒定时间比较**。
- **JWT 先验签名，再验其他**。
- **access 与 refresh 必须用 `type` 区分**。
- **refresh 必须轮换**，且要能检测重放。
- **重放时吊销整个族**，不是只吊销一张。
- **登录失败不区分原因**（防账号枚举）。
- **密钥走配置**，代码里出现的都该有 `noqa` 和注释说明。
- **时间要可注入**。

## 常见坑 / 排错

1. **`REPLAY_DETECTED` 在正常刷新时也触发** → 前端并发发了两次刷新请求，
   需要加刷新锁（或幂等窗口）。
2. **`WRONG_TYPE`** → 拿 refresh 去调业务接口了；检查 `expected` 参数。
3. **`BAD_SIGNATURE`** → 密钥不一致，或 token 被截断（注意 URL 里的 `.` 和换行）。
4. **测试里 token 不过期** → 没有注入 `clock`，用的是真实 `time.time()`。
5. **用户频繁被踢下线** → 网络重试导致重复刷新，先查是不是并发问题再关重放检测。
6. **`scrypt` 太慢** → 调低 `n`（但别低于 2^13），或换 `argon2id`（需装库）。

## 延伸练习

1. 加**刷新锁**：同一个 family 并发刷新时只放行第一个。
2. 实现 **access token 黑名单**（登出后 access 立即失效，不等 15 分钟过期）。
3. 加**设备信息**：`family` 里记设备名，让用户能看到"我的登录设备"并单独踢掉。
4. 把 `RefreshStore` 换成 Redis（复用第 35 章的 TTL 能力），支持多实例。

## 小结

鉴权的安全不在算法多复杂，而在**每一步都为最坏情况做过准备**：
密码泄露了怎么办（慢哈希）、token 泄露了怎么办（轮换 + 重放检测）、
账号被枚举了怎么办（统一错误）。
下一章（56）讲用户模块：仓储层、资料维护、验证码限流与账号安全。
