"""Identify the caller by the wallet address in the X-Wallet-Address header.

Temporary: an address is public on-chain, so this only checks "is this a known user", not "is it really them".
Proving ownership needs a signed challenge from the wallet's key pair.
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
