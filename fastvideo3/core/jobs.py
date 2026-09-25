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
    def __init__(self, db_url: str) -> None:
        self.engine = create_engine(db_url, future=True)
        metadata.create_all(self.engine)

    def submit(self, job: RunJob) -> None:
        import json

        with self.engine.begin() as conn:
            conn.execute(
                jobs_table.insert().values(
                    job_id=job.job_id,
                    run_id=job.run_id,
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
        with self.engine.begin() as conn:
            conn.execute(update(jobs_table).where(jobs_table.c.job_id == job_id).values(**values))

    def get(self, job_id: str) -> RunJob | None:
        import json

        with self.engine.connect() as conn:
            row = conn.execute(select(jobs_table).where(jobs_table.c.job_id == job_id)).first()
        if row is None:
            return None
        return RunJob(
            job_id=row.job_id,
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
                job_id=r.job_id,
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
