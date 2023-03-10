from create_keywords import dict_to_tensor, select_topk, create_clip_Da_dataset_from_saved, \
create_titles, create_clip_Da_dataset_from_saved_recipe1m_13m, create_clip_Da_dataset_from_saved_food101, create_titles



# Pretraining

# Create titles from constructed json files 

json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_albef_kw.json'
output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/vg_albef_ttl_kw.json'
tmp2 = create_titles(json_path, output_path=output_path)


json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_kw.json'
output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/sbu_ttl_kw.json'
tmp2 = create_titles(json_path, output_path=output_path)


json_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_kw.json'
output_path = '/data/mshukor/data/our_albef_data/clip_da/json_pretrain/coco_ttl_kw.json'
tmp2 = create_titles(json_path, output_path=output_path)


### Finetuning 
# ## recipe1m
image_root = None

dir_data = '/data/mshukor/data/recipe1m'
embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'
image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_train.json'

output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_train_ingr_kw.json'

split = 'train'

tmp = create_clip_Da_dataset_from_saved(dir_data, embeddings_path, image_embeddings_path, 
	k=15, output_path=output_path, split=split)





image_root = None

dir_data = '/data/mshukor/data/recipe1m'
embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'
image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_test.json'

output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_test_ingr_kw.json'

split = 'test'

tmp = create_clip_Da_dataset_from_saved(dir_data, embeddings_path, image_embeddings_path, 
	k=15, output_path=output_path, split=split)





image_root = None

dir_data = '/data/mshukor/data/recipe1m'
embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'
image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_val.json'

output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_val_ingr_kw.json'

split = 'val'

tmp = create_clip_Da_dataset_from_saved(dir_data, embeddings_path, image_embeddings_path, 
	k=15, output_path=output_path, split=split)









####### titles


dir_data = '/data/mshukor/data/recipe1m'
embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'
image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_train.json'

output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_train_titles_kw.json'

split = 'train'

tmp = create_clip_Da_dataset_from_saved(dir_data, embeddings_path, image_embeddings_path, 
	k=5, output_path=output_path, split=split)





image_root = None

dir_data = '/data/mshukor/data/recipe1m'
embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'
image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_test.json'

output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_test_titles_kw.json'

split = 'test'

tmp = create_clip_Da_dataset_from_saved(dir_data, embeddings_path, image_embeddings_path, 
	k=5, output_path=output_path, split=split)





image_root = None

dir_data = '/data/mshukor/data/recipe1m'
embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'
image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_val.json'

output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_val_titles_kw.json'

split = 'val'

tmp = create_clip_Da_dataset_from_saved(dir_data, embeddings_path, image_embeddings_path, 
	k=5, output_path=output_path, split=split)



### Recipe1m+



# image_root = None

# dir_data = '/data/mshukor/data/recipe1m'
# embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'
# image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_train.json'

# output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_recipe1m_13m_train_titles_kw.json'

# split = 'train'

# path_ids = '/data/mshukor/data/recipe1m/recipe1m_13m/ids.json'
# path_image_json = '/data/mshukor/data/recipe1m/recipe1m_13m/layer2+.json'

# tmp = create_clip_Da_dataset_from_saved_recipe1m_13m(path_ids, path_image_json, embeddings_path, image_embeddings_path, 
# 	k=5, output_path=output_path, split=split)



# image_root = None

# dir_data = '/data/mshukor/data/recipe1m'
# embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'
# image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_train.json'

# output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_recipe1m_13m_train_ingr_kw.json'

# split = 'train'

# path_ids = '/data/mshukor/data/recipe1m/recipe1m_13m/ids.json'
# path_image_json = '/data/mshukor/data/recipe1m/recipe1m_13m/layer2+.json'

# tmp = create_clip_Da_dataset_from_saved_recipe1m_13m(path_ids, path_image_json, embeddings_path, image_embeddings_path, 
# 	k=15, output_path=output_path, split=split)


# ########3 original ids

# image_root = None

# dir_data = '/data/mshukor/data/recipe1m'
# embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'
# image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_train.json'

# output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_recipe1m_13m_orig_val_ingr_kw.json'

# split = 'val'

# path_ids = '/data/mshukor/data/recipe1m/recipe1m_13m/original_ids.json'
# path_image_json = '/data/mshukor/data/recipe1m/recipe1m_13m/layer2+.json'

# tmp = create_clip_Da_dataset_from_saved_recipe1m_13m(path_ids, path_image_json, embeddings_path, image_embeddings_path, 
# 	k=15, output_path=output_path, split=split)



# image_root = None

# dir_data = '/data/mshukor/data/recipe1m'
# embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'
# image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_train.json'

# output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_recipe1m_13m_orig_test_ingr_kw.json'

# split = 'test'

# path_ids = '/data/mshukor/data/recipe1m/recipe1m_13m/original_ids.json'
# path_image_json = '/data/mshukor/data/recipe1m/recipe1m_13m/layer2+.json'

# tmp = create_clip_Da_dataset_from_saved_recipe1m_13m(path_ids, path_image_json, embeddings_path, image_embeddings_path, 
# 	k=15, output_path=output_path, split=split)





# image_root = None

# dir_data = '/data/mshukor/data/recipe1m'
# embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'
# image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_train.json'

# output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_recipe1m_13m_orig_val_titles_kw.json'

# split = 'val'

# path_ids = '/data/mshukor/data/recipe1m/recipe1m_13m/original_ids.json'
# path_image_json = '/data/mshukor/data/recipe1m/recipe1m_13m/layer2+.json'

# tmp = create_clip_Da_dataset_from_saved_recipe1m_13m(path_ids, path_image_json, embeddings_path, image_embeddings_path, 
# 	k=5, output_path=output_path, split=split)



# image_root = None

# dir_data = '/data/mshukor/data/recipe1m'
# embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'
# image_embeddings_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_train.json'

# output_path = '/data/mshukor/data/recipe1m/clip_da/layer1_recipe1m_13m_orig_test_titles_kw.json'

# split = 'test'

# path_ids = '/data/mshukor/data/recipe1m/recipe1m_13m/original_ids.json'
# path_image_json = '/data/mshukor/data/recipe1m/recipe1m_13m/layer2+.json'

# tmp = create_clip_Da_dataset_from_saved_recipe1m_13m(path_ids, path_image_json, embeddings_path, image_embeddings_path, 
# 	k=5, output_path=output_path, split=split)





################  Food 101



# dir_data = '/data/mshukor/data/recipe1m'
# embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'
# image_embeddings_path = '/data/mshukor/data/food101/clip_da/image_embeddings/food101.json'

# output_path = '/data/mshukor/data/food101/clip_da/layer1_food101_titles_kw.json'

# data_dir = '/data/mshukor/data/food101/food-101/images'

# tmp = create_clip_Da_dataset_from_saved_food101(data_dir=data_dir, embeddings_path=embeddings_path, 
# 	image_embeddings_path=image_embeddings_path, k=5, output_path=output_path)





# dir_data = '/data/mshukor/data/recipe1m'
# embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'
# image_embeddings_path = '/data/mshukor/data/food101/clip_da/image_embeddings/food101.json'

# output_path = '/data/mshukor/data/food101/clip_da/layer1_food101_ingr_kw.json'

# data_dir = '/data/mshukor/data/food101/food-101/images'

# tmp = create_clip_Da_dataset_from_saved_food101(data_dir=data_dir, embeddings_path=embeddings_path, 
# 	image_embeddings_path=image_embeddings_path, k=15, output_path=output_path)



################  isia 500f



embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/titles.json'
image_embeddings_path = '/data/mshukor/data/isia_food500/clip_da/image_embeddings/isia_food500.json'

output_path = '/data/mshukor/data/isia_food500/clip_da/layer1_isia_food500_titles_kw.json'

data_dir = '/data/mshukor/data/isia_food500/ISIA_Food500/images'

tmp = create_clip_Da_dataset_from_saved_food101(data_dir=data_dir, embeddings_path=embeddings_path, 
	image_embeddings_path=image_embeddings_path, k=5, output_path=output_path)





embeddings_path = '/data/mshukor/data/recipe1m/clip_da/embeddings/layer1_keywords.json'
image_embeddings_path = '/data/mshukor/data/isia_food500/clip_da/image_embeddings/isia_food500.json'

output_path = '/data/mshukor/data/isia_food500/clip_da/layer1_isia_food500_ingr_kw.json'

data_dir = '/data/mshukor/data/isia_food500/ISIA_Food500/images'

tmp = create_clip_Da_dataset_from_saved_food101(data_dir=data_dir, embeddings_path=embeddings_path, 
	image_embeddings_path=image_embeddings_path, k=15, output_path=output_path)






