from create_keywords import save_clip_embeddings




############################################## recipe1m



json_path = '/data/mshukor/data/recipe1m/clip_da/keywords/layer1_keywords.json'
output_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'

text_embed = save_clip_embeddings(json_path, output_path)




json_path = '/data/mshukor/data/recipe1m/clip_da/keywords/titles.json'
output_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'

text_embed = save_clip_embeddings(json_path, output_path)

