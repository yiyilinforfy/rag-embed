"""数据库连接：SQLAlchemy engine、Session 工厂、建表。

学习笔记
- engine 管理连接池；用 `with SessionLocal() as session:` 开一个会话，用完自动把连接还给连接池。
- pool_pre_ping：云数据库（Supabase）会断开空闲连接，借用连接前先 ping 一下，避免断线后的第一个请求报错。
- init_db() 做两件事：启用 pgvector 扩展（CREATE EXTENSION vector），然后按 models.py 建表。
- 没用 Alembic 这类迁移工具：create_all 只会创建"还不存在的表"，不会修改已有表的字段。
  以后要改表结构（比如改向量维度），要么手动 ALTER TABLE，要么引入 Alembic。
- 连接串格式：postgresql+psycopg://用户名:密码@主机:端口/库名
  本地是 Docker 里的 PostgreSQL（docker-compose.yml），线上是 Supabase 的 Session pooler（端口 5432，支持 IPv4）。
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings

# pre_ping: hosted databases (Supabase, RDS) drop idle connections; test before reuse instead of failing a request.
engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    # Import models so their tables are registered on Base.metadata before create_all.
    from app import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
