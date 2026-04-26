def run(name: str) -> dict:
    return {"name": name, "email": f"{name.lower().replace(' ', '.')}@example.com"}
