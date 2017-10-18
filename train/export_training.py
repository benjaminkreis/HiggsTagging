import numpy as np
import h5py
import os
import tarfile

def print_to_cpp(name, a):
    #put output in subdir for tarballing later
    if not os.path.isdir('keras_training'):
        os.mkdir('keras_training')

    f=open("keras_training/{}.h".format(name),"w")

    #meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {}\n".format(np.min(a)))
    f.write("//Max {}\n".format(np.max(a)))
    f.write("\n")
    
    #c++ variable 
    f.write("weight_t {}".format(name))
    for x in a.shape:
        f.write("[{}]".format(x))
    f.write(" = {")
    
    #fill c++ array.  
    #not including internal brackets for multidimensional case
    i=0;
    for x in np.nditer(a, order='C'):
        if i==0:
            f.write("{}".format(x))
        else:
            f.write(", {}".format(x))
        i=i+1
    f.write("}")
    f.close()




h5File = h5py.File('KERAS_check_best_model_weights.h5')

#print h5 contents
#for item in h5File.attrs.keys():
#    print(item + ":", h5File.attrs[item])

b1 = h5File['/fc1_relu/fc1_relu/bias:0'][()]
w1 = h5File['/fc1_relu/fc1_relu/kernel:0'][()]
b2 = h5File['/fc2_relu/fc2_relu/bias:0'][()]
w2 = h5File['/fc2_relu/fc2_relu/kernel:0'][()]
b3 = h5File['/fc3_relu/fc3_relu/bias:0'][()]
w3 = h5File['/fc3_relu/fc3_relu/kernel:0'][()]
b4 = h5File['/softmax/softmax/bias:0'][()]
w4 = h5File['/softmax/softmax/kernel:0'][()]

print_to_cpp("w1",w1)
print_to_cpp("b1",b1)
print_to_cpp("w2",w2)
print_to_cpp("b2",b2)
print_to_cpp("w3",w3)
print_to_cpp("b3",b3)
print_to_cpp("w4",w4)
print_to_cpp("b4",b4)

#tarball output
with tarfile.open('keras_training' + '.tar.gz', mode='w:gz') as archive:
    archive.add('keras_training', recursive=True)
