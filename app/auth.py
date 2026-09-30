"""鉴权（临时方案）：用请求头 X-Wallet-Address 里的钱包地址识别用户。

学习笔记
- 规则：没带地址或格式不对 → 401；地址不在 users 表或已停用 → 403；
  通过后返回 AuthUser，限流模块按这个地址计数。
- 前端从 Vuex store 里取当前钱包地址（wallets[walletIndex].address），放进请求头。
- 局限：钱包地址在链上是公开的，谁知道一个白名单地址都能冒充。
  所以这只能确认"是已知用户"，不能确认"请求的人真的拥有这个钱包"。
- 升级方向（签名验证）：后端发一个随机 nonce → 前端用钱包私钥签名 → 后端用公钥验证签名 → 签发带过期时间的 token。
  升级时只需要替换这个文件里"怎么确认地址"的部分，users 表和限流都不用改。
- 管理用户：scripts/users.py add / list / disable / enable。
"""

import re
from dataclasses import dataclass

from fastapi import Header, HTTPException
from sqlalchemy import select

from app.db import SessionLocal
from app.models import User

# VSYS addresses are base58 (no 0, O, I, l), 35 characters.
ADDRESS_PATTERN = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{30,40}$")


@dataclass(frozen=True)
class AuthUser:
    id: int
    address: str


def find_active_user(address: str) -> AuthUser | None:
    with SessionLocal() as session:
        user = session.scalar(select(User).where(User.address == address, User.is_active.is_(True)))
        return AuthUser(id=user.id, address=user.address) if user else None


def current_user(x_wallet_address: str = Header(default="")) -> AuthUser:
    address = x_wallet_address.strip()
    if not address:
        raise HTTPException(status_code=401, detail="缺少钱包地址，请先登录钱包")
    if not ADDRESS_PATTERN.match(address):
        raise HTTPException(status_code=401, detail="钱包地址格式不正确")
    user = find_active_user(address)
    if user is None:
        raise HTTPException(status_code=403, detail="当前钱包地址没有使用 VKube Assistant 的权限")
    return user
