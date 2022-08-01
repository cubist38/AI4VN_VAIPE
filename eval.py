from results.run import get_result

data_path = 'data/public_test'
get_result(data_path, output_path = './results/csv/result_test.csv')

# from evaluate.run import eval
# eval('results/csv/result_demo.csv', 'data/train.csv')