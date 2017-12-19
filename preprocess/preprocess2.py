
def preprocess2(file):
    f = open(file+'_pp.txt', 'r')
    g = open(file+'_pp2.txt', 'w')
    if file == 'test_data':
        for line in f:
            if line != ' <ALLCAPS>\n':
                g.write(line[18:]) #for test [18:] to remove initial useless part
    else:
        for line in f:
            if line != ' <ALLCAPS>\n':
                g.write(line)

def check_lines(file):
    f = open(file + '.txt', 'r')
    g = open(file + '_pp2.txt', 'r')
    count_f = len(f.readlines())
    count_g = len(g.readlines())
    print(count_f)
    print(count_g)

def main():
    name = 'train_pos_full'
    preprocess2(name)
    check_lines(name)

if __name__ == '__main__':
    main()


