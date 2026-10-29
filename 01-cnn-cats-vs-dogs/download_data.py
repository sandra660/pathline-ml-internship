"""Download the 4,000-image Cats vs. Dogs subset used in the notebook.

Source: Kaggle "Dogs vs. Cats" images, mirrored at github.com/laxmimerit/dog-cat-full-dataset.
Only the first 1,500 cats + 1,500 dogs of the train folder and 500 + 500 of the test folder
are fetched (about 100 MB), using a sparse git checkout. Requires git.
Result: data/train/{cats,dogs}/*.jpg and data/test/{cats,dogs}/*.jpg
"""
import os
import shutil
import subprocess

REPO = "https://github.com/laxmimerit/dog-cat-full-dataset.git"
TMP = "_dataset_tmp"
SUBSET = {"data/train/cats": 1500, "data/train/dogs": 1500, "data/test/cats": 500, "data/test/dogs": 500}


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def main():
    if os.path.isdir("data"):
        print("data/ already exists - nothing to do.")
        return
    shutil.rmtree(TMP, ignore_errors=True)
    run("git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1", REPO, TMP)
    all_files = run("git", "ls-tree", "-r", "--name-only", "HEAD", cwd=TMP).splitlines()
    wanted = []
    for folder, n in SUBSET.items():
        wanted += [f for f in all_files if f.startswith(folder + "/")][:n]
    run("git", "sparse-checkout", "init", "--no-cone", cwd=TMP)
    with open(os.path.join(TMP, ".git", "info", "sparse-checkout"), "w") as fh:
        fh.write("\n".join("/" + f for f in wanted) + "\n")
    print(f"Downloading {len(wanted)} images...")
    run("git", "checkout", "HEAD", cwd=TMP)
    shutil.move(os.path.join(TMP, "data"), "data")
    shutil.rmtree(TMP, ignore_errors=True)
    print("Done: images saved in data/")


if __name__ == "__main__":
    main()
