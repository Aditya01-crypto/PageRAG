from slowapi import Limiter
from slowapi.util import get_ipaddr
from pathlib import Path
limiter = Limiter(key_func=get_ipaddr)

UPLOAD_DIR = Path("uploaded_files")

