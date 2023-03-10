from create_keywords import extract_titles

 

# # ############################################## recipe1m


json_path = '/data/mshukor/data/recipe1m/recipe1M/layer1.json'
output_path = '/data/mshukor/data/recipe1m/clip_da/keywords/titles.json'

tmp = extract_titles(json_path, output_path)
 