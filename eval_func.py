import json
from utils.arguments_parse import args

def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        for line in lines:
            data = json.loads(line)
            args_list=[]
            for event in data['event_list']:
                event_type = event['event_type']
                args=[event_type+'_'+arg['argument'] for arg in event['arguments']]
                args_list.extend(args)
            sentences.append(args_list)
                    
        return sentences


def eval_function():

    true_data_list=load_data(args.test_path)
    pred_data_list=load_data('./output/result2.json')

    true_count=0
    pred_count=0
    corr_count=0

    for i in range(1498):
        true_data = true_data_list[i]
        pred_data = pred_data_list[i]
        corr = sum([1 for k in true_data if k in pred_data])

        true_count += len(true_data)
        pred_count += len(pred_data)
        corr_count += corr
    
    recall = corr_count / true_count
    precise = corr_count / pred_count

    print(recall)
    print(precise)

if __name__=='__main__':
    eval_function()
         
            