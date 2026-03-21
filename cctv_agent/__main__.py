import argparse
import socket


def main():
    parser = argparse.ArgumentParser(prog="cctv-agent")
    parser.add_argument("--server",  required=True, help="Central server IP")
    parser.add_argument("--node-id", default=None,  help="Node ID (default: machine hostname)")
    parser.add_argument("--config",  default=None,  help="Path to config.yaml")
    parser.add_argument("streams",   nargs="*",     help="Optional stream URLs or video directory")
    args = parser.parse_args()

    # Always use hostname as node ID unless user explicitly overrides
    # This means zero config needed — just run and connect
    node_id = args.node_id or socket.gethostname()

    print(f"[Agent] Starting node '{node_id}' -> server {args.server}")

    from .main import run, collect_video_sources

    stream_urls = collect_video_sources(args.streams) if args.streams else []

    run(
        server=args.server,
        node_id=node_id,
        config_path=args.config,
        stream_urls=stream_urls,
    )


if __name__ == "__main__":
    main()