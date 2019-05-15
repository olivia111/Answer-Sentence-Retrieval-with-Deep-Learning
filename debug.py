import tensorflow as tf





def debug(ops, list_loader):
    init_ops = [tf.tables_initializer(),
                tf.global_variables_initializer()]

    with tf.Session() as sess:
        # train_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        sess.run(init_ops)
        list_loader.start(sess)

        results = sess.run(ops)
        for i in results:
            print(i)
            print("")

    exit(1)