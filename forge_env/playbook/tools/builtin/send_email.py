def run(to: str, subject: str, body: str) -> dict:
    return {"status": "queued", "to": to, "subject": subject, "preview": body[:80]}
