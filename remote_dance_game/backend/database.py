import aiosqlite
import json
import os
from typing import Optional, List, Dict, Any
from config import DB_PATH


async def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS dances (
                dance_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                duration_ms INTEGER DEFAULT 0,
                skeleton_format TEXT DEFAULT 'blazepose_33',
                preview_mode TEXT DEFAULT 'local_video',
                difficulty TEXT DEFAULT 'medium',
                mirror_mode INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                num_frames INTEGER DEFAULT 0,
                num_events INTEGER DEFAULT 0,
                video_path TEXT,
                audio_path TEXT,
                dance_dir TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                dance_id TEXT,
                status TEXT DEFAULT 'queued',
                progress INTEGER DEFAULT 0,
                stage TEXT DEFAULT '',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                dance_id TEXT NOT NULL,
                status TEXT DEFAULT 'created',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results (
                session_id TEXT PRIMARY KEY,
                dance_id TEXT NOT NULL,
                total_score INTEGER DEFAULT 0,
                max_combo INTEGER DEFAULT 0,
                grade_counts TEXT DEFAULT '{}',
                hold_results TEXT DEFAULT '[]',
                accuracy_arms REAL DEFAULT 0,
                accuracy_legs REAL DEFAULT 0,
                accuracy_torso REAL DEFAULT 0,
                timeline_scores TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        await db.commit()


async def get_db():
    return await aiosqlite.connect(DB_PATH)


async def insert_dance(data: Dict[str, Any]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO dances
               (dance_id, title, version, duration_ms, skeleton_format, preview_mode,
                difficulty, mirror_mode, created_at, num_frames, num_events, video_path, audio_path, dance_dir)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (data["dance_id"], data["title"], data.get("version", 1),
             data.get("duration_ms", 0), data.get("skeleton_format", "blazepose_33"),
             data.get("preview_mode", "local_video"), data.get("difficulty", "medium"),
             1 if data.get("mirror_mode", True) else 0, data["created_at"],
             data.get("num_frames", 0), data.get("num_events", 0),
             data.get("video_path"), data.get("audio_path"), data["dance_dir"])
        )
        await db.commit()


async def get_dance(dance_id: str) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM dances WHERE dance_id = ?", (dance_id,))
        row = await cursor.fetchone()
        if row:
            d = dict(row)
            d["mirror_mode"] = bool(d["mirror_mode"])
            return d
    return None


async def list_dances() -> List[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM dances ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def delete_dance(dance_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM dances WHERE dance_id = ?", (dance_id,))
        await db.commit()


async def insert_job(data: Dict[str, Any]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO jobs
               (job_id, dance_id, status, progress, stage, error, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (data["job_id"], data.get("dance_id"), data.get("status", "queued"),
             data.get("progress", 0), data.get("stage", ""),
             data.get("error"), data["created_at"], data["updated_at"])
        )
        await db.commit()


async def update_job(job_id: str, **kwargs):
    async with aiosqlite.connect(DB_PATH) as db:
        sets = ", ".join(f"{k} = ?" for k in kwargs.keys())
        vals = list(kwargs.values()) + [job_id]
        await db.execute(f"UPDATE jobs SET {sets} WHERE job_id = ?", vals)
        await db.commit()


async def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def insert_session(data: Dict[str, Any]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO sessions (session_id, dance_id, status, created_at)
               VALUES (?, ?, ?, ?)""",
            (data["session_id"], data["dance_id"], data.get("status", "created"), data["created_at"])
        )
        await db.commit()


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def update_session(session_id: str, **kwargs):
    async with aiosqlite.connect(DB_PATH) as db:
        sets = ", ".join(f"{k} = ?" for k in kwargs.keys())
        vals = list(kwargs.values()) + [session_id]
        await db.execute(f"UPDATE sessions SET {sets} WHERE session_id = ?", vals)
        await db.commit()


async def insert_result(data: Dict[str, Any]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO results
               (session_id, dance_id, total_score, max_combo, grade_counts, hold_results,
                accuracy_arms, accuracy_legs, accuracy_torso, timeline_scores, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (data["session_id"], data["dance_id"], data.get("total_score", 0),
             data.get("max_combo", 0), json.dumps(data.get("grade_counts", {})),
             json.dumps(data.get("hold_results", [])),
             data.get("accuracy_arms", 0), data.get("accuracy_legs", 0),
             data.get("accuracy_torso", 0), json.dumps(data.get("timeline_scores", [])),
             data["created_at"])
        )
        await db.commit()


async def get_result(session_id: str) -> Optional[Dict[str, Any]]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM results WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()
        if row:
            d = dict(row)
            d["grade_counts"] = json.loads(d["grade_counts"])
            d["hold_results"] = json.loads(d["hold_results"])
            d["timeline_scores"] = json.loads(d["timeline_scores"])
            return d
    return None


async def set_setting(key: str, value: Any):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)""",
            (key, json.dumps(value)),
        )
        await db.commit()


async def get_setting(key: str, default: Any = None) -> Any:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = await cursor.fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return default


async def get_all_settings() -> Dict[str, Any]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
        out: Dict[str, Any] = {}
        for row in rows:
            key = row["key"]
            try:
                out[key] = json.loads(row["value"])
            except Exception:
                out[key] = row["value"]
        return out
