import tarfile
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Unzip file')
    parser.add_argument("--file", '-f', type=str, required=True)

    args = parser.parse_args()
    # open file
    file = tarfile.open(os.path.abspath(args.file))
    # extracting file
    file.extractall(os.path.abspath('./dataset'))

    file.close()


if __name__ == "__main__":
    main()