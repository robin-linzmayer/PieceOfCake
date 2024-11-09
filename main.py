import argparse
from piece_of_cake_game import PieceOfCakeGame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", "-d", type=int, default=5,
                        help="tolerance for the cake size")
    parser.add_argument("--seed", "-s", type=int, default=2, help="Seed used by random number generator")
    parser.add_argument(
        "--requests", "-rq", help="Use the given requests, if no requests are given, Generate requests using L"
    )
    parser.add_argument("--scale", "-sc", default=50, help="Scale")
    parser.add_argument("--no_gui", "-ng", action="store_true", help="Disable GUI")
    parser.add_argument("--log_path", default="log", help="Directory path to dump log files, filepath if "
                                                          "disable_logging is false")
    parser.add_argument("--disable_logging", action="store_true", help="Disable Logging, log_path becomes path to file")
    parser.add_argument("--disable_timeout", action="store_true", help="Disable timeouts for player code")
    parser.add_argument("--player", "-p", default="d", help="Specifying player")
    args = parser.parse_args()

    if args.disable_logging:
        if args.log_path == "log":
            args.log_path = "results.log"

    root = None
    app = PieceOfCakeGame(args, root)

