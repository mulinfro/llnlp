
from bert import modeling

def create_bert_model(bert_config, is_training, input_ids, input_mask, seg_ids, use_one_hot_embeddings):
    model = modeling.BertConfig(
        config = bert_config,
        is_training = is_training,
        input_ids = input_ids,
        input_mask = input_mask,
        token_type_ids = seg_ids,
        use_one_hot_embeddings = use_one_hot_embeddings
    )

