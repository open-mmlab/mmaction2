with open('/data/junbeom/repo/mmaction2/lab/data/ucfTrainTestlist/testlist01.txt', 'r') as f :
    data = f.readlines()

data_split = []
for data_line in data :
    data_split.append(data_line.split('/')[0])

find_plt_target = []
latest = ''
for i, line in enumerate(data_split) :
    if latest == '' or line != latest :
        latest = line
        find_plt_target.append(str(i))

sample_num = 3783

print(find_plt_target, end='\t')
with open('/data/junbeom/repo/mmaction2/lab/data/ann_for_plt_ucf101_test_01.txt', 'w') as f :
    f.write('\n'.join(find_plt_target))
    f.write('\n' + str(sample_num))
print('finish')