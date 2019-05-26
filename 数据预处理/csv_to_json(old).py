import os
import json
import csv
from collections import namedtuple

# 创建json模板
jsonData = {
    # 五个基本字段
    'info': {},
    
    'images': [],
    
    'licenses': [],
    
    'annotations': [],
    
    'categories': []
}

# 类名id映射表
classname_to_id = {"0": 0,"1": 1,"2": 2,"3": 3,"4": 4,"5": 5,"6": 6,"7": 7,
                   "8": 8,"9": 9,"10": 10,"11": 11,"12": 12,"13": 13,"14": 14,
                   "15": 15,"16": 16,"17": 17,"18": 18,"19": 19,"20": 20}
				
# 初始化categories字段
categories = []

# 添加categories字段元素
for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            categories.append(category)

# 更新json数据categories字段
jsonData['categories'] = categories
print(jsonData['categories'])

# 初始化images字段
images = []

# 初始化annotations字段
annotations = []

filename = "1.csv"
with open(filename) as f:
    f_csv = csv.reader(f)
    headings = next(f_csv)
    Row = namedtuple('Row', headings)
    for i,r in enumerate(f_csv):
        row = Row(*r)
        
        # 初始化临时存储
        image = {}
        annotation = {}
        
        # 添加images字段元素
        image['license'] = 1
        image['file_name'] = row.filename
        image['height'] = 1600
        image['width'] = 3200
        image['id'] = i
        images.append(image)
        
        # 添加annotations字段元素
        annotation['segmentation'] = []
        annotation['area'] = 10000
        annotation['iscrowd'] = 0
        annotation['image_id'] = i
        annotation['bbox'] = [int(row.xmin), int(row.ymin), (int(row.xmax) - int(row.xmin)), (int(row.ymax) - int(row.ymin))]
        annotation['category_id'] = int(row.name)
        annotations.append(annotation)

# 更新json数据images字段
jsonData["images"] = images
# 更新json数据annotations字段
jsonData["annotations"] = annotations

with open('data.json', 'w') as f:
    json.dump(jsonData, f, indent=2)  #indent表示格式化的空格数