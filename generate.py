import os
import frontmatter
from pathlib import Path
import shutil
import yaml

# Directories
BASE_DIR = Path(__file__).parent.absolute()
TAGS_DIR = Path(BASE_DIR, "_tags")
if not os.path.exists(TAGS_DIR):
    os.makedirs(TAGS_DIR)

def get_unique_tags(dir):
    """Get set of unique tags from all
    markdown posts in a given directory.
    The tags are parsed from each posts's
    frontmatter metadata."""
    # Get markdown files (recursively)
    fps = []
    for path in Path(dir).rglob('*.md'):
        fps.append(path)

    # Get unique tags
    unique_tags = set()
    for fp in fps:
        fm = frontmatter.load(fp)
        if "tags" in fm:
            tags = fm["tags"].split(" ")
            unique_tags.update(tags)

    return unique_tags


def generate_tag_pages():
    """Generate tag markdown pages."""
    # Get collections
    with open(Path("_config.yml"), 'r') as f:
        config = yaml.safe_load(f)
    collections = list(config["collections"].keys())

    # Collect unique tags
    tags = set()
    for collection in collections:
        collection_dir = f"_{collection}"
        tags.update(get_unique_tags(dir=collection_dir))

    # Reset tags dir
    shutil.rmtree(TAGS_DIR)
    if not os.path.exists(TAGS_DIR):
        os.makedirs(TAGS_DIR)

    # Create tag pages
    for tag in tags:
        with open(f"{TAGS_DIR}/{tag}.md", 'a') as f:
            fm_str = f"---\nlayout: tag\ntitle: 'Tag: {tag}'\ntag-name: {tag}\n---\n"
            f.write(fm_str)

    print(f"{len(tags)} tag pages created")


if __name__ == "__main__":

    # Generate tag pages
    generate_tag_pages()
