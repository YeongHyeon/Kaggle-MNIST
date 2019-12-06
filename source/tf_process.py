import os, inspect

import tensorflow as tf
import numpy as np

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint', sess.graph)

    iteration = 0

    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries], \
                feed_dict={neuralnet.x:x_tr, neuralnet.y:y_tr, neuralnet.batch_size:x_tr.shape[0]}, \
                options=run_options, run_metadata=run_metadata)
            loss, accuracy = sess.run([neuralnet.loss, neuralnet.accuracy], \
                feed_dict={neuralnet.x:x_tr, neuralnet.y:y_tr, neuralnet.batch_size:x_tr.shape[0]})
            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  Loss:%.3f, Acc:%.3f" \
            %(epoch, epochs, iteration, loss, accuracy))
        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        print("\nRestoring parameters")
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")

    fsubmit = open("submission.csv", "w")
    fsubmit.write("ImageId,Label\n")
    cntid = 1

    confusion_matrix = np.zeros((dataset.num_class, dataset.num_class), np.int32)
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.
        class_score = sess.run(neuralnet.score, \
            feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})

        label, logit = np.argmax(y_te[0]), np.argmax(class_score)
        confusion_matrix[label, logit] += 1

        fsubmit.write("%d,%d\n" %(cntid, logit))
        cntid += 1

        if(terminator): break
    fsubmit.close()
