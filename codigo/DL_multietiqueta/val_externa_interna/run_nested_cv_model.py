from pathlib import Path

from dl_nested_cv_common import default_input_profile_for_model, run_model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input-profile', choices=['all', 'focused', 'densenet_light'], default='')
    parser.add_argument('--run-tag', default='')
    parser.add_argument('--output-root', default='')
    args = parser.parse_args()

    input_profile = args.input_profile or default_input_profile_for_model(args.model)
    output_root = Path(args.output_root) if args.output_root else None
    run_model(args.model, input_profile=input_profile, run_tag=args.run_tag, output_root=output_root)


if __name__ == '__main__':
    main()
