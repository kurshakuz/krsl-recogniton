import os
import numpy as np
import tensorflow as tf
import argparse

def load_data_nested(dirname):
    if dirname[-1]!='/':
        dirname=dirname+'/'
    listfile=os.listdir(dirname)
    X = []
    Y = []
    # print(listfile)
    # print(dirname)
    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text
            numbers=[]
            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()]
                for i in range(len(numbers),25200):
                    numbers.extend([0.000]) 
            landmark_frame=[]
            row=0
            for i in range(0,70):
                landmark_frame.extend(numbers[row:row+84])
                row += 84
            landmark_frame=np.array(landmark_frame)
            landmark_frame=landmark_frame.reshape(-1,84)
            X.append(np.array(landmark_frame))
            Y.append(wordname)
    X=np.array(X)
    Y=np.array(Y)
    # print(Y)
    x_train = X
    x_train=np.array(x_train)
    return x_train,Y

def load_data(dirname):
    if dirname[-1] != '/':
        dirname=dirname+'/'
    listfile=os.listdir(dirname)
    X = []
    Y = []
    for text in listfile:
        textname = dirname + text
        numbers=[]
        with open(textname, mode = 'r') as t:
            numbers = [float(num) for num in t.read().split()]
            for i in range(len(numbers),25200):
                numbers.extend([0.000])
        landmark_frame=[]
        row=0
        for i in range(0,70):
            landmark_frame.extend(numbers[row:row+84])
            row += 84
        landmark_frame=np.array(landmark_frame)
        landmark_frame=landmark_frame.reshape(-1,84)
        X.append(np.array(landmark_frame))
        Y.append(text)
    X = np.array(X)
    Y = np.array(Y)
    # print(Y)
    x_train = X
    x_train=np.array(x_train)
    return x_train,Y

def load_label():
    listfile=[]
    with open("sign-prediction/label.txt",mode='r') as l:
        listfile=[i for i in l.read().split()]
    label = {}
    count = 1
    for l in listfile:
        if "_" in l:
            continue
        label[l] = count
        count += 1
    return label

def parse_video_and_generate_files(input_data_path, output_data_path):
    comp='bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \mediapipe/examples/desktop/hand_tracking:hand_tracking_cpu'
    cmd='GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/hand_tracking/hand_tracking_cpu \--calculator_graph_config_file=mediapipe/graphs/hand_tracking/hand_tracking_desktop_live.pbtxt'
    listfile=os.listdir(input_data_path)

    if not(os.path.isdir(output_data_path+"Relative/")):
        os.mkdir(output_data_path+"Relative/")
    if not(os.path.isdir(output_data_path+"Absolute/")):
        os.mkdir(output_data_path+"Absolute/")

    for file in listfile:
        if not(os.path.isdir(input_data_path+file)):
            continue
        word = file + "/"
        fullfilename=os.listdir(input_data_path+word)
        if not(os.path.isdir(output_data_path+"_"+word)):
            os.mkdir(output_data_path+"_"+word)
        if not(os.path.isdir(output_data_path+"Relative/"+word)):
            os.mkdir(output_data_path+"Relative/"+word)
        if not(os.path.isdir(output_data_path+"Absolute/"+word)):
            os.mkdir(output_data_path+"Absolute/"+word)
        os.system(comp)
        outputfilelist = os.listdir(output_data_path + '_' + word)
        for mp4list in fullfilename:
            if ".DS_Store" in mp4list:
                continue         
            inputfilen='   --input_video_path='+input_data_path+word+mp4list
            outputfilen='   --output_video_path='+output_data_path+'_'+word+mp4list
            cmdret = cmd + inputfilen + outputfilen
            os.system(cmdret)
    
def recogintion(files_nested, processed_data_path):
    output_dir = processed_data_path
    if files_nested:
        x_test, Y = load_data_nested(output_dir)
    else:
        x_test, Y = load_data(output_dir)

    # new_model = tf.keras.models.load_model('sign-prediction/model.h5')
    new_model = tf.keras.models.load_model('sign-prediction/model_isolated_rel.h5')

    
    new_model.summary()

    labels=load_label()
    xhat = x_test
    yhat = new_model.predict(xhat)
    predictions = np.array([np.argmax(pred) for pred in yhat])
    rev_labels = dict(zip(list(labels.values()), list(labels.keys())))
    
    # print(yhat)
    # print(Y)
    # print(predictions)
    # print(rev_labels)

    # for s, i in enumerate(predictions):
    #     print("Ground truth: ", Y[s])
    #     print("Predicted sign: ", rev_labels[i])
    #     print("Confidence: ", round(yhat[0][i]*100, 2), "%")
    #     print("Confidence: ", yhat[0][i])
    
    s = 0
    txtpath = processed_data_path + "result.txt" 
    with open(txtpath, "w") as f:
        for i in predictions:
            f.write(Y[s])
            f.write(" ")
            f.write(rev_labels[i])
            f.write("\n")
            s += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Sign language with Mediapipe')
    parser.add_argument("--input_data_path",help=" ")
    parser.add_argument("--output_data_path",help=" ")
    # parser.add_argument("--processed_data_path",help=" ")
    # parser.add_argument("--files_nested",help="1 or 0")
    args = parser.parse_args()

    input_data_path = args.input_data_path
    output_data_path = args.output_data_path
    # processed_data_path = args.processed_data_path
    processed_data_path = output_data_path + "/Relative/"
    files_nested = True

    parse_video_and_generate_files(input_data_path, output_data_path)
    recogintion(files_nested, processed_data_path)