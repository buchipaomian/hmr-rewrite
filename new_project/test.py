import pandas as pd


json_name = "json_body/body3DScene_00027642.json"
k = pd.read_json(json_name)
k = k.to_dict()
print(k["bodies"][0]["joints19"])
joint = k["bodies"][0]["joints19"]
joint_result = []
for i in range(len(joint)):
    if i % 4 != 3:
        joint_result.append(joint[i])        
print(joint_result)