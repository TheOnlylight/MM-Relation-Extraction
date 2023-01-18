import fire
def data_from(k, d):
    image_name = k.replace('.pkl','')
    image_path = os.path.join(
        '/data2/private/zhouhantao/data/datatest/standard/mod_std/standard/image',
        image_name
    )
    ocr_path = os.path.join(
        '/data2/private/zhouhantao/data/datatest/standard/standard/image_ocr_result',
        k,
    )
    with open(ocr_path, 'rb') as f:
        ocr = pickle.load(f)
    ocr_token = [x[1][0] for x in ocr[0]]
    ocr_bbox = [x[0] for x in ocr[0]]
    def post_process(data, pair):
        ans = [
        (
            data.stdenz2ocr[item[0]],
            data.cid2ocr[item[1]]
        )
            for item in pair
    ]
        return ans
    relation_pair = list(set(post_process(d, d.pairs)))
    return {
        'ocr-path': ocr_path,
        'relation-pair': relation_pair,
        'ocr-token': ocr_token,
        'ocr-bbox': ocr_bbox,
        'image_path': image_path,
        'image': Image.open(image_path)
    }
def generate_ds():
    for k in valid_keys:
        yield data_from(k, data[k])
        
from datasets import Dataset

def main(
    output_dir: str = './output_data',
    input_dir: str = './input_data',
    
):
    ds = Dataset.from_generator(generate_ds)
    ds.push_to_hub('Hantao/ChemReactionImageRE')
    
if __name__ == '__main__':
  fire.Fire(main)