"""Background job to clean up rendered page images older than 24 hours."""
import os
import time
import shutil
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

async def cleanup_old_pages(ttl_seconds: int = 24 * 3600):
    """Remove directories in logs/pages/ that haven't been accessed for ttl_seconds."""
    pages_dir = Path("logs/pages")
    if not pages_dir.exists():
        return
        
    now = time.time()
    deleted_count = 0
    
    try:
        # Loop through child directories in logs/pages/
        for p in pages_dir.iterdir():
            if p.is_dir():
                # Get the last modification/access time of the directory
                stat = p.stat()
                mtime = max(stat.st_mtime, stat.st_atime)
                
                # If older than TTL, purge it
                if now - mtime > ttl_seconds:
                    logger.info(f"Purging expired page assets for contract ID: {p.name}")
                    shutil.rmtree(p)
                    deleted_count += 1
        if deleted_count > 0:
            logger.info(f"Page asset cleanup complete. Purged {deleted_count} expired contract directories.")
    except Exception as e:
        logger.error(f"Error executing page assets cleanup: {e}")

async def start_periodic_cleanup_job(interval_seconds: int = 12 * 3600):
    """Run the cleanup job periodically in the background."""
    logger.info("Initializing periodic page asset cleanup background worker.")
    while True:
        await asyncio.sleep(interval_seconds)
        await cleanup_old_pages()
