from sklearn.model_selection import train_test_split
import os
import argparse 

def moveFiles(x_path, paths, sets):
    '''paths: [train_path, vali_path, test_path]
    sets: [train_set, vali_set, test_set]'''
    for each in sets[0]:
        shutil.copyfile(os.path.join(x_path,each), os.path.join(paths[0], each))

    for each in sets[1]:
        shutil.copyfile(os.path.join(x_path,each), os.path.join(paths[1], each))

    for each in sets[2]:
        shutil.copyfile(os.path.join(x_path,each), os.path.join(paths[2], each))
        
    print('train set: {}, validation set: {}, test set: {}'.format(len(sets[0]), len(sets[1]), len(sets[2])))
    
    
def matchY(set_x):
    set_y = []
    for each in set_x:
        y = each.split('_combined.tiff')[0]+'_ilastik_s2_Probabilities.tiff'
        set_y.append(y) 
    return(set_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/mnt/davinci/temp/xx_data/IMC/Dice-XMDB/data/BRCA2/')
    args = parser.parse_args()
    
    
    path1 = args.data
    x_path = os.path.join(path1, 'combined2_image')
    y_path = os.path.join(path1, 'probability')

    x_files = os.listdir(x_path)
    y_files = os.listdir(y_path)

    train_set_x, vali_set_x = train_test_split(x_files, test_size=0.4, random_state=33)
    vali_set_x, test_set_x = train_test_split(vali_set_x, test_size=0.5, random_state=33)

    train_set_y = matchY(train_set_x)
    vali_set_y = matchY(vali_set_x)
    test_set_y = matchY(test_set_x)

    train_x_path = os.path.join(path1, 'train', 'combined2_image')
    train_y_path = os.path.join(path1, 'train', 'probability')
    vali_x_path = os.path.join(path1, 'vali', 'combined2_image')
    vali_y_path = os.path.join(path1, 'vali', 'probability')
    test_x_path = os.path.join(path1, 'test', 'combined2_image')
    test_y_path = os.path.join(path1, 'test', 'probability')

    for each in [train_x_path, train_y_path, vali_x_path, vali_y_path, test_x_path, test_y_path]:
        if not os.path.exists(each):
            os.makedirs(each)

    x_paths = [train_x_path, vali_x_path, test_x_path]
    x_sets = [train_set_x, vali_set_x, test_set_x]

    moveFiles(x_path, x_paths, x_sets)

    y_paths = [train_y_path, vali_y_path, test_y_path]
    y_sets = [train_set_y, vali_set_y, test_set_y]

    moveFiles(y_path, y_paths, y_sets)