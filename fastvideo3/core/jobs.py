"""异步任务台账：SQLAlchemy Core + SQLite/PostgreSQL（TDD §3.5）。

续跑时凭 job_id 先查旧任务状态再决定是否重提交 —— 不重复花钱。
"""

from __future__ import annotations

import time

from sqlalchemy import (
    Column,
    Float,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    update,
)

from fastvideo3.core.schema import JobStatus, RunJob

metadata = MetaData()

jobs_table = Table(
    "jobs",
    metadata,
    Column("job_id", String(64), primary_key=True),
    Column("run_id", String(64), index=True),
    Column("capability", String(16)),
    Column("provider", String(32)),
    Column("status", String(16)),
    Column("params_json", Text),
    Column("output_uri", Text, nullable=True),
    Column("error", Text, nullable=True),
    Column("submitted_at", Float),
    Column("finished_at", Float, nullable=True),
)


class JobStore:
    """异步任务台账。

    两个"类防"（都是真踩出来的，不是假想）：

    1. **主键按 run 命名空间化**。`job_id` 形如 `scene_s1` / `i2v_shot_1_1_0`，天生只
       在单个 run 内唯一；而它是表的主键 —— 于是第二条 run 用同样的镜头号会撞上同一行：
       要么 `IntegrityError`，要么 `get()` 拿回**别的 run** 的任务状态，续跑时把别人
       的 task_id 当成自己的轮询并把它的产物下载进本 run。存储层用 `run_id#job_id`
       做键、对调用方透明（读写都还是传/收裸 `job_id`）。
    2. **`update()` 见不到行就插一条**。此前只调 `update()` 不调 `submit()` 的模块
       （`m4_scene` 就是这样）台账永远是空的，`get()` 恒返回 None，于是"跳过已成功
       任务""在途任务续跑"两条防重复付费的保护全是死代码。现在 update 本身就是
       upsert，这类"忘了插行"再也静默不掉。
    """

    def __init__(self, db_url: str, run_id: str = "") -> None:
        self.engine = create_engine(db_url, future=True)
        self.run_id = run_id
        metadata.create_all(self.engine)

    def _key(self, job_id: str) -> str:
        if not self.run_id or "#" in job_id:
            return job_id
        return f"{self.run_id}#{job_id}"

    def _plain(self, key: str) -> str:
        pref = f"{self.run_id}#"
        return key[len(pref):] if self.run_id and key.startswith(pref) else key

    def submit(self, job: RunJob) -> None:
        import json

        with self.engine.begin() as conn:
            key = self._key(job.job_id)
            if conn.execute(
                select(jobs_table.c.job_id).where(jobs_table.c.job_id == key)
            ).first() is not None:
                # 已有行：只补元信息，不重置状态（重置会把 SUCCEEDED 抹成 PENDING → 重复提交）
                conn.execute(
                    update(jobs_table)
                    .where(jobs_table.c.job_id == key)
                    .values(params_json=json.dumps(job.params, ensure_ascii=False))
                )
                return
            conn.execute(
                jobs_table.insert().values(
                    job_id=key,
                    run_id=job.run_id or self.run_id,
                    capability=job.capability,
                    provider=job.provider,
                    status=job.status.value,
                    params_json=json.dumps(job.params, ensure_ascii=False),
                    output_uri=job.output_uri,
                    error=job.error,
                    submitted_at=time.time(),
                    finished_at=None,
                )
            )

    def update(
        self,
        job_id: str,
        status: JobStatus | None = None,
        output_uri: str | None = None,
        error: str | None = None,
    ) -> None:
        values: dict = {}
        if status is not None:
            values["status"] = status.value
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                values["finished_at"] = time.time()
        if output_uri is not None:
            values["output_uri"] = output_uri
        if error is not None:
            values["error"] = error
        if not values:
            return
        key = self._key(job_id)
        with self.engine.begin() as conn:
            hit = conn.execute(
                update(jobs_table).where(jobs_table.c.job_id == key).values(**values)
            ).rowcount
            if not hit:
                # 行不存在 → 插入。**这是"只 update 不 submit"的兜底**：没有它，
                # 模块忘记插行时台账永远为空，"跳过已提交任务"静默失效，重启就重复付费。
                conn.execute(
                    jobs_table.insert().values(
                        job_id=key,
                        run_id=self.run_id,
                        capability="",
                        provider="",
                        status=values.get("status", JobStatus.PENDING.value),
                        params_json="{}",
                        output_uri=values.get("output_uri"),
                        error=values.get("error"),
                        submitted_at=time.time(),
                        finished_at=values.get("finished_at"),
                    )
                )

    def get(self, job_id: str) -> RunJob | None:
        import json

        with self.engine.connect() as conn:
            row = conn.execute(select(jobs_table).where(jobs_table.c.job_id == self._key(job_id))).first()
        if row is None:
            return None
        return RunJob(
            job_id=self._plain(row.job_id),
            run_id=row.run_id or "",
            capability=row.capability or "",
            provider=row.provider or "",
            params=json.loads(row.params_json or "{}"),
            status=JobStatus(row.status or "pending"),
            output_uri=row.output_uri,
            error=row.error,
        )

    def list_open(self, run_id: str) -> list[RunJob]:
        """该 run 中尚未成功的任务（续跑检查用）。"""
        import json

        with self.engine.connect() as conn:
            rows = conn.execute(
                select(jobs_table).where(
                    jobs_table.c.run_id == run_id,
                    jobs_table.c.status.in_(["pending", "running"]),
                )
            ).all()
        return [
            RunJob(
                job_id=self._plain(r.job_id),
                run_id=r.run_id or "",
                capability=r.capability or "",
                provider=r.provider or "",
                params=json.loads(r.params_json or "{}"),
                status=JobStatus(r.status or "pending"),
                output_uri=r.output_uri,
                error=r.error,
            )
            for r in rows
        ]

    def dispose(self) -> None:
        self.engine.dispose()
