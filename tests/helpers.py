from pathlib import Path

from app.data import load_math500

FIXTURE = Path(__file__).parent / "fixtures" / "math500_sample.jsonl"
QS = load_math500(FIXTURE)


async def collect(it):
    return [e async for e in it]
