import re


pattern = '"img_path": "'
with open('vid.json', 'r') as fin:
    with open('vid2.json', 'w', encoding='utf8') as fout:
        for line in fin:
            if re.findall(pattern, line):
                line = re.sub(r'ILSVRC2015_VID_.+\\\\ILSVRC2015_.+\\\\', '', line)
            fout.write(line)

