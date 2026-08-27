"""
Project-ABC Brain -- Entry Point
Boot sequence: HW detect -> Soul load -> Orchestrator -> API server
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.hardware.hw_detector import scan_hardware
from brain.orchestrator import Orchestrator
from brain.utils.config import load_config
from brain.utils.logger import configure_logging, get_logger

log = get_logger(__name__)

_restart_requested: bool = False


async def run(config: dict, dry_run: bool = False, open_browser: bool = False) -> None:
    hw_cfg = config.get("brain", {})
    configure_logging(hw_cfg.get("log_file", "data/brain.log"), hw_cfg.get("log_level", "INFO"))

    # Suppress WinError 10054 "connection forcibly closed" — this fires whenever a
    # browser tab refreshes or a WebSocket client disconnects on Windows. It is
    # benign (normal TCP teardown) but asyncio's ProactorEventLoop logs it as ERROR.
    loop = asyncio.get_event_loop()
    _orig_exception_handler = loop.get_exception_handler()

    def _exception_handler(loop, ctx):
        exc = ctx.get("exception")
        if isinstance(exc, ConnectionResetError):
            return  # swallow — browser disconnect noise on Windows
        msg = ctx.get("message", "")
        if "connection" in msg.lower() and "lost" in msg.lower():
            return  # swallow _call_connection_lost variants
        if _orig_exception_handler:
            _orig_exception_handler(loop, ctx)
        else:
            loop.default_exception_handler(ctx)

    loop.set_exception_handler(_exception_handler)

    log.info("=" * 50)
    log.info("  Project-ABC Brain Starting...")
    log.info("=" * 50)

    api_cfg = config.get("api", {})

    # Step 1: Start API + open browser immediately so user sees loading screen
    # while the rest of the brain initialises in the background.
    api_port = None
    _api_server = None
    if not dry_run and api_cfg.get("enabled", True):
        api_port, _api_server = await _start_api_early(api_cfg)
        if open_browser and api_port:
            import webbrowser
            url = f"http://localhost:{api_port}/display"
            log.info(f"Opening browser now (loading screen): {url}")
            webbrowser.open(url)
            # CRITICAL: yield the event loop for 1.5 s so the server can fully
            # serve display.html AND loading.gif to the browser before
            # scan_hardware() blocks the event loop for ~8 s.
            await asyncio.sleep(1.5)

    # Boot status emitter -- api is already running at this point
    from api.ws_api import set_boot_status

    # Step 2: Hardware detection
    mock = hw_cfg.get("mock_hardware", False) or os.getenv("MOCK_HARDWARE", "false").lower() == "true"
    set_boot_status("[1/4] Scanning hardware (camera / mic / GPIO)...")
    await asyncio.sleep(0.4)   # yield so WS can deliver this status before blocking
    log.info("Scanning hardware...")
    hw = scan_hardware(mock_mode=mock)

    # Report what was found
    parts = []
    if getattr(hw, "camera_available", False):  parts.append("Camera")
    if getattr(hw, "mic_available",    False):  parts.append("Mic")
    if getattr(hw, "motor_available",  False):  parts.append("Motors")
    hw_summary = ", ".join(parts) if parts else "Mock hardware"
    set_boot_status(f"[1/4] Hardware ready -- {hw_summary}")
    await asyncio.sleep(0.4)

    # Step 3: Build orchestrator (loads soul, creates drivers, pre-warms models)
    set_boot_status("[2/4] Loading soul, memory & world model...")
    await asyncio.sleep(0.4)
    log.info("Loading soul and memory...")
    orch = Orchestrator(config=config, hw=hw)

    if dry_run:
        orch._build_agents()
        log.info("Dry run mode -- all agents initialized, exiting without starting.")
        print("\n[OK] Brain initialized successfully (dry run)")
        print(f"     Hardware: {orch._build_hw_summary()}")
        print(f"     Soul loaded: {bool(orch._soul.soul)}")
        print(f"     Agents ({len(orch._agents)}): {', '.join(orch._agents.keys())}")
        return

    # Step 4: Start all agents
    set_boot_status("[3/4] Launching agents (STT, TTS, Vision, Emotion, Reasoning)...")
    await asyncio.sleep(0.4)
    log.info("Starting agents...")
    await orch.start()

    # Step 5: Wire orchestrator into already-running API (or start API if not started)
    set_boot_status("[4/4] Wiring orchestrator into API...")
    await asyncio.sleep(0.4)
    if api_port:
        from api.rest_api import set_orchestrator
        from api.ws_api import set_orchestrator as ws_set_orch
        set_orchestrator(orch)
        ws_set_orch(orch)
        log.info("Orchestrator wired into API -- brain fully online")
    elif api_cfg.get("enabled", True):
        log.info(f"Starting API at http://localhost:{api_cfg.get('port', 8080)}")
        await _start_api(orch, api_cfg, open_browser=open_browser)

    set_boot_status("Brain fully online -- all systems go!")
    await asyncio.sleep(0.4)
    log.info("Brain fully online.")

    from brain.agents.base_agent import AgentMessage, MessageType

    # Signal display to leave boot screen and show eye expressions
    await orch.inject(AgentMessage(
        type=MessageType.ACTION_DISPLAY,
        source="main",
        data={"boot_complete": True},
        priority=1,
    ))

    # Short pause so the eye GIF renders before speaking
    await asyncio.sleep(1.5)

    # Boot announcement
    await orch.inject(AgentMessage(
        type=MessageType.ACTION_SPEAK,
        source="main",
        data={"text": "I am awake. Hello, world."},
        priority=1,
    ))

    # Run until Ctrl+C or SIGTERM.
    # On Windows, asyncio.run() converts KeyboardInterrupt into CancelledError on the
    # main task — we just await here and let that propagation do the work.
    # SIGTERM is handled via loop.add_signal_handler on Linux; on Windows it is
    # unavailable so we skip it gracefully.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop():
        if not stop_event.is_set():
            log.info("Shutdown requested — stopping brain...")
            loop.call_soon_threadsafe(stop_event.set)

    # SIGTERM via asyncio (Linux/Mac only)
    try:
        loop.add_signal_handler(signal.SIGTERM, _request_stop)
    except (NotImplementedError, OSError):
        pass

    # SIGINT (Ctrl+C) via signal.signal — works reliably on Windows ProactorEventLoop
    # where loop.add_signal_handler is not supported.
    import signal as _signal
    _signal.signal(_signal.SIGINT, lambda s, f: _request_stop())

    # Register restart callback with the API so web UI restart doesn't use os.execv
    try:
        from api.rest_api import set_restart_callback
        def _do_web_restart():
            global _restart_requested
            _restart_requested = True
            _request_stop()
        set_restart_callback(_do_web_restart)
    except ImportError:
        pass

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        log.info("Shutting down brain...")
        # Stop uvicorn first so it releases the port and its task exits cleanly.
        # This prevents asyncio.run() from hanging while waiting for the api-server task.
        if _api_server is not None:
            _api_server.should_exit = True
            await asyncio.sleep(0.3)  # give uvicorn a moment to drain connections
        await orch.stop()
        log.info("Brain offline. Goodbye.")


def _find_free_port(preferred: int, retries: int = 10) -> int:
    """Return preferred port if free, otherwise find next available one."""
    import socket
    for port in range(preferred, preferred + retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
            return port
        except OSError:
            continue
    raise OSError(f"No free port found in range {preferred}-{preferred + retries}")


async def _start_api_early(api_cfg: dict):
    """Start uvicorn with the FastAPI app but WITHOUT an orchestrator wired in.
    The display page shows loading.gif; orchestrator is injected later via set_orchestrator().
    Returns (port, server) or (None, None) on failure."""
    try:
        import uvicorn
        from api.rest_api import app
        from api.ws_api import ws_endpoint

        # Register WS route once
        existing = {r.path for r in app.routes if hasattr(r, "path")}
        if "/ws" not in existing:
            app.add_api_websocket_route("/ws", ws_endpoint)

        preferred = api_cfg.get("port", 8080)
        host = api_cfg.get("host", "0.0.0.0")
        port = _find_free_port(preferred)

        cfg = uvicorn.Config(app, host=host, port=port,
                             log_level="warning", loop="none", lifespan="off")
        server = uvicorn.Server(cfg)

        async def _serve():
            try:
                await server.serve()
            except (SystemExit, OSError, asyncio.CancelledError, KeyboardInterrupt):
                pass

        asyncio.create_task(_serve(), name="api-server")

        # Copy loading.gif into static dir so /static/loading.gif is served
        _ensure_loading_gif()

        # Wait until uvicorn has actually bound the socket (up to 5 s)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while not server.started and loop.time() < deadline:
            await asyncio.sleep(0.1)

        if server.started:
            log.info(f"API server ready on http://localhost:{port} (loading screen)")
        else:
            log.warning("API server may not have started in time -- opening browser anyway")

        return port, server
    except Exception as e:
        log.error(f"Early API start failed: {e}")
        return None, None


def _ensure_loading_gif() -> None:
    """Make loading.gif available at /loading.gif (static dir)."""
    import shutil
    from pathlib import Path
    src = Path("assets/loading.gif")
    dst_dir = Path("api/static")
    dst = dst_dir / "loading.gif"
    if src.exists() and not dst.exists():
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            log.info("Copied loading.gif -> api/static/loading.gif")
        except Exception as e:
            log.warning(f"Could not copy loading.gif: {e}")


async def _start_api(orch, api_cfg: dict, open_browser: bool = False) -> None:
    import uvicorn
    from api.rest_api import app, set_orchestrator
    from api.ws_api import set_orchestrator as ws_set_orch, ws_endpoint

    set_orchestrator(orch)
    ws_set_orch(orch)
    # Guard against duplicate WebSocket route registration
    existing_routes = {r.path for r in app.routes if hasattr(r, "path")}
    if "/ws" not in existing_routes:
        app.add_api_websocket_route("/ws", ws_endpoint)

    preferred = api_cfg.get("port", 8080)
    host = api_cfg.get("host", "0.0.0.0")

    try:
        port = _find_free_port(preferred)
    except OSError as e:
        log.error(f"API server cannot start: {e} -- brain continues without API")
        return

    if port != preferred:
        log.warning(f"Port {preferred} in use -- using {port} instead")

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        loop="none",
        lifespan="off",
    )
    server = uvicorn.Server(config)

    async def _serve_safe():
        try:
            await server.serve()
        except (SystemExit, OSError) as e:
            log.error(f"API server stopped: {e}")
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass  # Normal on brain shutdown

    asyncio.create_task(_serve_safe(), name="api-server")
    log.info(f"API server starting on http://localhost:{port}")

    if open_browser:
        await asyncio.sleep(0.5)
        import webbrowser
        url = f"http://localhost:{port}/display"
        log.info(f"Opening display in browser: {url}")
        webbrowser.open(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project-ABC Robotic Brain")
    parser.add_argument("--config",    default="config/config.yaml", help="Config file path")
    parser.add_argument("--dry-run",   action="store_true", help="Initialize and exit without starting")
    parser.add_argument("--mock",      action="store_true", help="Force mock hardware mode")
    parser.add_argument("--laptop",    action="store_true", help="Laptop mode: use webcam/mic/speakers + open display in browser")
    parser.add_argument("--no-speech", action="store_true", help="Disable TTS output")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mock:
        cfg.setdefault("brain", {})["mock_hardware"] = True

    if args.laptop:
        cfg.setdefault("brain", {})["laptop_mode"] = True
        cfg.setdefault("api",   {})["enabled"] = True
        cfg.setdefault("display", {})["type"] = "web"

    if args.no_speech:
        cfg.setdefault("audio", {}).setdefault("output", {})["tts_engine"] = "none"

    global _restart_requested
    open_browser = args.laptop  # only open browser on first start

    while True:
        _restart_requested = False
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run(cfg, dry_run=args.dry_run, open_browser=open_browser))
        except (KeyboardInterrupt, SystemExit):
            if not _restart_requested:
                _restart_requested = False  # Ctrl+C = clean exit, not restart
        finally:
            try:
                # Cancel every remaining task and wait max 3 s — then exit regardless.
                # asyncio.run()'s built-in _cancel_all_tasks has no timeout and hangs
                # on Windows when C-extension threads (camera, audio) don't respond.
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.wait(pending, timeout=3.0))
            except BaseException:  # catches KeyboardInterrupt on Windows ProactorEventLoop
                pass
            finally:
                try:
                    loop.close()
                except BaseException:
                    pass

        if not _restart_requested:
            break

        log.info("Restarting brain...")
        cfg = load_config(args.config)   # reload config so changes take effect
        open_browser = False              # don't reopen browser on restart


if __name__ == "__main__":
    main()