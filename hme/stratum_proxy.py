"""
Minimal Stratum V1 TCP proxy for Bitaxe when the i7 pool host is offline.

Listens on LAN :3333 and forwards to an upstream solo pool (default public-pool.io).
This restores a *local primary* path without requiring a fully-synced bitcoind.

Usage:
  python -m hme.stratum_proxy
  # or
  python -m hme proxy
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from typing import Optional

log = logging.getLogger("hme.proxy")

DEFAULT_UPSTREAM_HOST = os.environ.get("HME_UPSTREAM_HOST", "public-pool.io")
DEFAULT_UPSTREAM_PORT = int(os.environ.get("HME_UPSTREAM_PORT", "21496"))
DEFAULT_LISTEN_HOST = os.environ.get("HME_PROXY_HOST", "0.0.0.0")
DEFAULT_LISTEN_PORT = int(os.environ.get("HME_PROXY_PORT", "3333"))


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, label: str) -> None:
    try:
        while not reader.at_eof():
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError, OSError) as e:
        log.debug("%s pipe closed: %s", label, e)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def handle_miner(
    miner_reader: asyncio.StreamReader,
    miner_writer: asyncio.StreamWriter,
    upstream_host: str,
    upstream_port: int,
) -> None:
    peer = miner_writer.get_extra_info("peername")
    log.info("miner connected %s → %s:%s", peer, upstream_host, upstream_port)
    try:
        up_reader, up_writer = await asyncio.open_connection(upstream_host, upstream_port)
    except OSError as e:
        log.error("upstream connect failed %s:%s: %s", upstream_host, upstream_port, e)
        miner_writer.close()
        await miner_writer.wait_closed()
        return

    t1 = asyncio.create_task(pipe(miner_reader, up_writer, "miner→up"))
    t2 = asyncio.create_task(pipe(up_reader, miner_writer, "up→miner"))
    done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    log.info("miner disconnected %s", peer)


async def run_proxy(
    listen_host: str,
    listen_port: int,
    upstream_host: str,
    upstream_port: int,
) -> None:
    server = await asyncio.start_server(
        lambda r, w: handle_miner(r, w, upstream_host, upstream_port),
        listen_host,
        listen_port,
    )
    addrs = ", ".join(str(s.getsockname()) for s in server.sockets or [])
    log.info("stratum proxy listening %s  upstream=%s:%s", addrs, upstream_host, upstream_port)
    async with server:
        await server.serve_forever()


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="HME Stratum V1 proxy (local primary repair)")
    p.add_argument("--listen-host", default=DEFAULT_LISTEN_HOST)
    p.add_argument("--listen-port", type=int, default=DEFAULT_LISTEN_PORT)
    p.add_argument("--upstream-host", default=DEFAULT_UPSTREAM_HOST)
    p.add_argument("--upstream-port", type=int, default=DEFAULT_UPSTREAM_PORT)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop(*_):
        log.info("shutting down proxy…")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(
            run_proxy(args.listen_host, args.listen_port, args.upstream_host, args.upstream_port)
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
