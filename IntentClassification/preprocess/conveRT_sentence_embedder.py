
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as tfhub


TRAIN_MLP_FILE = "../data-train/train_mlp_full.txt"

sess = None
# Load the tensorflow hub module into a fresh session.

if sess is not None:
    sess.close()

sess = tf.InteractiveSession(graph=tf.Graph())
module = tfhub.Module("http://models.poly-ai.com/convert/v1/model.tar.gz")


text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
encoding_tensor = module(text_placeholder)
encoding_dim = int(encoding_tensor.shape[1])
print(f"ConveRT encodes text to {encoding_dim}-dimensional vectors")

sess.run(tf.tables_initializer())
sess.run(tf.global_variables_initializer())


def encode(text):
    """Encode the given texts to the encoding space."""
    return sess.run(encoding_tensor, feed_dict={text_placeholder: text})[0]


import pandas as pd


train_data = pd.read_csv(TRAIN_MLP_FILE, delimiter='\t')
sentence_embeddings = { }

for description in train_data['desc']:
 sentence_embeddings[description] = encode([description])