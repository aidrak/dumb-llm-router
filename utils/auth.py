from fastapi import Header, HTTPException


def validate_api_key(authorization: str = Header(None)) -> str:
    """Simple API key validation - accept any reasonable looking key"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if authorization.startswith("Bearer "):
        api_key = authorization[7:]
    else:
        api_key = authorization

    if len(api_key) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key format")

    return api_key
