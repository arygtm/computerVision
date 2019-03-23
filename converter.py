import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile

def graphdef_to_pbtxt(filename):
  with gfile.FastGFile(filename,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='a')
    tf.train.write_graph(graph_def, 'pbtxt/', 'protobuf.pbtxt', as_text=True)

  return


graphdef_to_pbtxt("./ExploreOpencvDnn/models/faster_rcnn_resnet101_kitti/frozen_inference_graph.pb")
print("Graph written out")
