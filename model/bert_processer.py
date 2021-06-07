
from bert import modeling

def create_bert_output_layer(bert_config, is_training, input_ids, input_mask, seg_ids, layer_type, use_one_hot_embeddings):
    model = modeling.BertModel(
        config = bert_config,
        is_training = is_training,
        input_ids = input_ids,
        input_mask = input_mask,
        token_type_ids = seg_ids,
        use_one_hot_embeddings = use_one_hot_embeddings
    )

    if layer_type.tolower() == "pooled":
        bert_output_layer = model.get_pooled_output()
    elif layer_type.tolower() == "last2":
        bert_output_layer = model.get_all_encoder_layers()[-2]
    else:
        bert_output_layer = model.get_sequence_output()


    return bert_output_layer
    

