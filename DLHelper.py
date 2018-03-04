import h5py, cv2
import csv, time, os.path
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle
from sklearn import model_selection as ms

# function to process a single image
def processImage(prefix, size, gtReader, proc_type=None, is_lisa=False, class_match=None):
    images = []
    labels = []

    for row in gtReader:
        if is_lisa:
            params = {"name": row[0], \
                    "box": (int(row[3]), int(row[5]), int(row[2]), int(row[4])), \
                    "label": class_match[row[1]] if row[1] in class_match.keys() else None}
            if params['label'] is None: # No such class
                print(row[1])
                continue
        else:
            params = {"name": row[0], \
                    "box": (int(row[4]), int(row[6]), int(row[3]), int(row[5])), \
                    "label": int(row[7])}

        image = cv2.imread(prefix + params["name"])
        if image.shape[2] != 3: # Gray?
            print(params["name"])

        # image = image[...,::-1] # BGR to RGB
        image = image[params["box"][0]:params["box"][1], params["box"][2]:params["box"][3]] # Crop the ROI
        image = cv2.resize(image, size) # Resize images 
        if proc_type is None:
            pass
        elif proc_type == "clahe":
            # lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) # BGR to Lab space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) # BGR to Lab space
            tmp = np.zeros((lab.shape[0],lab.shape[1]), dtype=lab.dtype)
            tmp[:,:] = lab[:,:,0] # Get the light channel of LAB space
            clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(4,4)) # Create CLAHE object
            light = clahe.apply(tmp) # Apply to the light channel
            lab[:,:,0] = light # Merge back
            # image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) # LAB to RGB
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) # LAB to RGB
        elif proc_type == "1sigma" or proc_type == "2sigma":
            # R, G, B = image[:,:,0], image[:,:,1], image[:,:,2] # RGB channels
            B, G, R = image[:,:,0], image[:,:,1], image[:,:,2]
            if proc_type == "1sigma":
                param = 1
            else: # "2sigma"
                param = 2
            # image[:,:,0] = cv2.normalize(R, None, R.mean() - param * R.std(), R.mean() + param * R.std(), cv2.NORM_MINMAX)
            image[:,:,0] = cv2.normalize(B, None, B.mean() - param * B.std(), B.mean() + param * B.std(), cv2.NORM_MINMAX)
            image[:,:,1] = cv2.normalize(G, None, G.mean() - param * G.std(), G.mean() + param * G.std(), cv2.NORM_MINMAX)
            # image[:,:,2] = cv2.normalize(B, None, B.mean() - param * B.std(), B.mean() + param * B.std(), cv2.NORM_MINMAX)
            image[:,:,2] = cv2.normalize(R, None, R.mean() - param * R.std(), R.mean() + param * R.std(), cv2.NORM_MINMAX)

        if not hasattr(image, 'shape'):
            print(image)
            print(params["name"])

        images.append(image) # Already uint8
        labels.append(params["label"])

    return images, labels

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrafficSigns_GT(rootpath, size, process=None, training=True):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 43 classes
    if training:
        for c in range(0,43):
            prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
            gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
            next(gtReader) # skip header
            # loop over all images in current annotations file
            imgs, lbls = processImage(prefix, size, gtReader, process)
            images = images + imgs
            labels = labels + lbls
            gtFile.close()
    else:
        gtFile = open(rootpath + "/../../GT-final_test.csv") # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        imgs, lbls = processImage(rootpath + '/', size, gtReader, process)
        images = images + imgs
        labels = labels + lbls
        gtFile.close()

    return images, labels

def readTrafficSigns_Belgium(rootpath, size, process=None, training=True):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all classes
    for c in range(0,62):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        imgs, lbls = processImage(prefix, size, gtReader, process)
        images = images + imgs
        labels = labels + lbls
        gtFile.close()

    return images, labels

def readLISACategories(rootpath):
    # Read categories
    f = open("{}categories.txt".format(rootpath))
    content = f.readlines()

    # Get categories
    count = 0
    class_match = {}
    for line in content:
        splitted = (line.strip().split(': ')[-1]).split(', ')
        for c in splitted:
            if c == "thruTrafficMergeLeft":
                class_match[c] = class_match["thruMergeLeft"] # Duplicated
                continue
            class_match[c] = count
            count += 1
    class_num = len(class_match.keys()) - 1
    f.close()
    return class_match, class_num

def readTrafficSigns_LISA(rootpath, size, process=None, training=True):
    class_match, class_num = readLISACategories(rootpath)

    images = []
    labels = []

    # All folder names
    folders = []
    folders += ["aiua120214-{}".format(i) for i in range(0, 3)]
    folders += ["aiua120306-{}".format(i) for i in range(0, 2)]
    folders += ["vid{}".format(i) for i in range(0, 12)]

    # Read all annotations
    for folder in folders:
        folder = rootpath + folder
        under = os.listdir(folder)
        for u in under:
            if u.startswith("frame"):
                folder = '/'.join([folder, u])
                break
        annotations = folder + "/frameAnnotations.csv"
        gtFile = open(annotations)
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        imgs, lbls = processImage(folder + "/", size, gtReader, process, True, class_match)
        images = images + imgs
        labels = labels + lbls
        gtFile.close()

    trainImages, testImages, trainLabels, testLabels = ms.train_test_split(images, labels, test_size=0.2, random_state=542)

    return trainImages, trainLabels, testImages, testLabels, class_num

def getDirFuncClassNum(root, dataset="GT"):
    train_dir, test_dir, readTrafficSigns = None, None, None
    class_num = -1
    if dataset == "GT":
        root = '/'.join([root, "GTSRB"])
        train_dir = '/'.join([root, "Final_Training/Images"])
        test_dir = '/'.join([root, "Final_Test/Images"])
        readTrafficSigns = readTrafficSigns_GT
        class_num = 43
    elif dataset == "Belgium":
        root = '/'.join([root, "BelgiumTSC"])
        train_dir = '/'.join([root, "Training"])
        test_dir = '/'.join([root, "Testing"])
        readTrafficSigns = readTrafficSigns_Belgium
        class_num = 62
    elif dataset == "LISA":
        root = '/'.join([root, "LISA"])
        train_dir = None
        test_dir = None
        readTrafficSigns = readTrafficSigns_LISA
        class_num = 46 # 1 duplicated, 47
    else:
        raise Exception("No such dataset!")

    return root, train_dir, test_dir, readTrafficSigns, class_num


def getImageSets(root, resize_size, dataset="GT", preprocessing=None, printing=True):
    root, train_dir, test_dir, readTrafficSigns, class_num = getDirFuncClassNum(root, dataset)
    trainImages, trainLabels, testImages, testLabels = None, None, None, None

    preprocessing = preprocessing if (preprocessing is not None) else "original"

    ## If pickle file exists, read the file
    if os.path.isfile(root + "/processed_images_{}_{}_{}_{}.pkl".format(resize_size[0], resize_size[1], dataset, preprocessing)):
        f = open(root + "/processed_images_{}_{}_{}_{}.pkl".format(resize_size[0], resize_size[1], dataset, preprocessing), 'rb')
        trainImages = cPickle.load(f, encoding="latin1")
        trainLabels = cPickle.load(f, encoding="latin1")
        testImages = cPickle.load(f, encoding="latin1")
        testLabels = cPickle.load(f, encoding="latin1")
        f.close()
    ## Else, read images and write to the pickle file
    else:
        print("Process {} dataset with {} and size {}, saved to {}.".format(dataset, preprocessing, resize_size, root))
        start = time.time()
        if dataset == "GT" or dataset == "Belgium":
            trainImages, trainLabels = readTrafficSigns(train_dir, resize_size, preprocessing, True)
            testImages, testLabels = readTrafficSigns(test_dir, resize_size, preprocessing, False)
        else: # LISA
            trainImages, trainLabels, testImages, testLabels, class_num = readTrafficSigns(root, resize_size, preprocessing)
            print(class_num)
        print("Training and testing Image preprocessing finished in {:.2f} seconds".format(time.time() - start))
        
        f = open(root + "/processed_images_{}_{}_{}_{}.pkl".format(resize_size[0], resize_size[1], dataset, preprocessing), 'wb')

        for obj in [trainImages, trainLabels, testImages, testLabels]:
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    if printing:
        print(trainImages[42].shape)
        plt.imshow(trainImages[42])
        plt.show()

        print(testImages[21].shape)
        plt.imshow(testImages[21])
        plt.show()

    return root, trainImages, trainLabels, testImages, testLabels, class_num

def init_h5py(filename, epoch_num, max_total_batch):
    f = h5py.File(filename, 'w')
        
    try:
        # config group for some common params
        config = f.create_group('config')
        config.attrs["total_epochs"] = epoch_num

        # cost group for training and validation cost
        cost = f.create_group('cost')
        loss = cost.create_dataset('loss', (epoch_num,))
        loss.attrs['time_markers'] = 'epoch_freq'
        loss.attrs['epoch_freq'] = 1
        train = cost.create_dataset('train', (max_total_batch,)) # Set size to maximum theoretical value
        train.attrs['time_markers'] = 'minibatch'

        # time group for batch and epoch time
        t = f.create_group('time')
        loss = t.create_dataset('loss', (epoch_num,))
        train = t.create_group('train')
        start_time = train.create_dataset("start_time", (1,), dtype='float64')
        start_time.attrs['units'] = 'seconds'
        end_time = train.create_dataset("end_time", (1,), dtype='float64')
        end_time.attrs['units'] = 'seconds'
        train_batch = t.create_dataset('train_batch', (max_total_batch,)) # Same as above

        # accuracy group for training and validation accuracy
        acc = f.create_group('accuracy')
        acc_v = acc.create_dataset('valid', (epoch_num,))
        acc_v.attrs['time_markers'] = 'epoch_freq'
        acc_v.attrs['epoch_freq'] = 1
        acc_t = acc.create_dataset('train', (max_total_batch,))
        acc_t.attrs['time_markers'] = 'minibatch'

        # Mark which batches are the end of an epoch
        time_markers = f.create_group('time_markers')
        time_markers.attrs['epochs_complete'] = epoch_num
        train_batch = time_markers.create_dataset('minibatch', (epoch_num,))

        # Inference accuracy
        infer = f.create_group('infer_acc')
        infer_acc = infer.create_dataset('accuracy', (1,))

    except Exception as e:
        f.close() # Avoid hdf5 runtime error or os error
        raise e # Catch the exception to close the file, then raise it to stop the program

    return f

def create_dir(current_dir, subs, model, devices):
    for sub in subs:
        path = os.path.join(current_dir, sub)
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, model)
        if not os.path.exists(path):
            os.makedirs(path)
        
        temp_path = path
        for device in devices:
            path = os.path.join(temp_path, device)
            if not os.path.exists(path):
                os.makedirs(path)

if __name__ == '__main__':
    root = "/Users/moderato/Downloads/"
    resize_size = (48, 48)
    # print(getImageSets(root, resize_size, dataset="LISA", process=None, printing=True))
