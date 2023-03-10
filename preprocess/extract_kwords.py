from create_keywords import extract_keywords

 

# # ############################################## recipe1m

json_path = '/data/mshukor/data/recipe1m/recipe1M/layer1.json'
output_path = '/data/mshukor/data/recipe1m/clip_da/keywords/layer1_keywords.json'
thresh = 1
key = 'ingredients'

objs, atts, rels = extract_keywords(json_path, extract_rel=False, extract_att=False, 
	output_path=output_path, thresh=thresh, key=key)



 