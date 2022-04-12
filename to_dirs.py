import csv
import shutil
from pathlib import Path

root = Path('dataset')
with open(root / 'Test.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_class = row['ClassId']
        class_path = root / 'Test_dirs' / img_class
        img_path = root / Path(row['Path'])

        class_path.mkdir(exist_ok=True)
        shutil.copy(img_path, class_path / img_path.name)