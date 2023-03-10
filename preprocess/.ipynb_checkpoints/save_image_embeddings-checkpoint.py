from create_keywords import save_image_embeddings, save_image_embeddings_recipe1m_13m, save_image_embeddings_food101, save_image_embeddings_imagenet






dir_data = '/data/mshukor/data/recipe1m'
output_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_test.json'
split='test'

tmp = save_image_embeddings(dir_data=dir_data, output_path=output_path, split=split)

dir_data = '/data/mshukor/data/recipe1m'
output_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_val.json'
split='val'

tmp = save_image_embeddings(dir_data=dir_data, output_path=output_path, split=split)


dir_data = '/data/mshukor/data/recipe1m'
output_path = '/data/mshukor/data/recipe1m/clip_da/image_embeddings/recipe1m_train.json'
split='train'

tmp = save_image_embeddings(dir_data=dir_data, output_path=output_path, split=split)


#### Recipe1m+



# output_path = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_train.json'
# split='train'
# path_ids = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/recipe1m_13m/ids.json'
# path_image_json = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/recipe1m_13m/layer2+.json'
# image_root = '/gpfsdswork/dataset/Recipe1M+/images_recipe1M+'

# tmp = save_image_embeddings_recipe1m_13m(path_ids, path_image_json, output_path=output_path, 
#     image_root=image_root, split=split)



# output_path = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_orig_val.json'
# split='val'
# path_ids = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/recipe1m_13m/original_ids.json'
# path_image_json = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/recipe1m_13m/layer2+.json'
# image_root = '/gpfsdswork/dataset/Recipe1M+/images_recipe1M+'

# tmp = save_image_embeddings_recipe1m_13m(path_ids, path_image_json, output_path=output_path, 
#     image_root=image_root, split=split)



# output_path = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/clip_da/image_embeddings/recipe1m_13m_orig_test.json'
# split='test'
# path_ids = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/recipe1m_13m/original_ids.json'
# path_image_json = '/gpfsscratch/rech/dyf/ugz83ue/data/recipe1m/recipe1m_13m/layer2+.json'
# image_root = '/gpfsdswork/dataset/Recipe1M+/images_recipe1M+'

# tmp = save_image_embeddings_recipe1m_13m(path_ids, path_image_json, output_path=output_path, 
#     image_root=image_root, split=split)






