import argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["textgen","codegen"], required=True)
    args, rest = ap.parse_known_args()

    if args.task == "textgen":
        from train_textgen import main as run
    else:
        from train_codegen import main as run

    sys.argv = [sys.argv[0]] + rest
    return run()

if __name__ == "__main__":
    raise SystemExit(main())
