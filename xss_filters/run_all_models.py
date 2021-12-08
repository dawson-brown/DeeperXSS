from lstm_sigmoid import lstm_model as sigmoid_model
from lstm_softmax import lstm_model as softmax_model
from lstm_sequence import lstm_model as sequence_model
from lstm_random_embed import lstm_model as random_embed_lstm
from data.tokenizer import URLTokens, JSToken


models = [
    ('sigmoid', sigmoid_model),
    ('softmax', softmax_model),
    ('sequence', sequence_model),
    ('random', random_embed_lstm)
]


def print_avg_results_to_file(model_name: str, results: dict) -> None:
    
    precision = 0
    recall = 0
    f1 = 0
    accuracy = 0
    folds = len(results)

    for result in results:
        precision += result['precision']
        recall += result['recall']
        f1 += result['f1']
        accuracy += result['accuracy']

    with open(f'{model_name}.model', 'a') as result_file:
        result_file.write(f'Precision: {precision/folds}\n')
        result_file.write(f'Recall: {recall/folds}\n')
        result_file.write(f'F1: {f1/folds}\n')
        result_file.write(f'Accuracy: {accuracy/folds}\n')



def main():
    
    for name, model in models:
        results = model("value")
        print_avg_results_to_file(f'{name}-value', results)

    for name, model in models:
        results = model("type")
        print_avg_results_to_file(f'{name}-type', results)


if __name__ == '__main__':
    main()