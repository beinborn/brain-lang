"""Random usefull functions.

"""

from itertools import combinations
import numpy as np

def get_folds(block_list, test_ratio=0.25):
  """Given the list of blocks splits them into all possible test and train sets

  :param block_list:
  :param test_ratio:
  :return:
  """
  total_number_of_blocks = len(block_list)
  test_size = int(total_number_of_blocks * test_ratio)
  train_size = total_number_of_blocks - test_size
  train_blocks = list([list(a) for a in combinations(block_list,train_size)])

  test_blocks = []
  for i in np.arange(len(train_blocks)):
    test_block = [x for x in block_list if x not in train_blocks[i]]
    test_blocks.append(test_block)


  return list(zip(train_blocks, test_blocks))