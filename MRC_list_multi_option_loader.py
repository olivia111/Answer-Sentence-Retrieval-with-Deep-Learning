import tensorflow as tf

# from data_preprocess.list_text_process.tags import useful_tags

'''
glove_path
sent_length_limit
data: 
'''


useful_tags = ["<ol>", "<ul>", "</ol>", "</ul>", "<listsep>", "<li>", "</li>",
                "<h>", "<h1>", "<h2>", "<h3>", "<h4>", "<h5>", "<h6>", "<h7>",
                "</h>", "</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</h7>","<th>","<td>","<tr>","<table>","<tbody>","<thead>","</th>","</td>","</tr>","</table>","</tbody>","</thead>"]
chunk_index_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "g0", "g1", "g2", "g3", "g5", "x"]
unknown_chunk_index_token = "x"


class MRCListMultiOptionLoader():

    def __init__(self, config, name):

        if name == "train" or name == "test":
            print("build %s data" % name)
        else:
            raise Exception("name needs to be train or test")


        # phrase_tags
        self.name = name
        self.config = config
        self.voc_table = None
        self.label_table = None
        self.data = None
        self.name = name
        self.input_dict = {}
        self._load_data()

    @classmethod
    def get_num_html_tags(cls):
        return len(useful_tags)

    @classmethod
    def get_num_special_tokens(cls):
        '''
        patternpad
        patterngo
        patterneos
        patternunk
        '''
        return 2

    def get_num_chars(self):
        # including special char and common char
        return len(self.char_id2voc)

    def _get_special_token_id(self, s):

        if s in self.voc_dic:
            return self.voc_dic[s]
        else:
            raise Exception("special token id not found in voc")

    def _get_special_char_id(self, c):
        if c in self.char_voc_dic:
            return self.char_voc_dic[c]
        else:
            raise Exception("special char id not found in char voc %s" % c)

    def _load_data(self):
        print("load data...")
        self.voc_dic = None
        self.id2voc = None
        self._load_vocabulary()
        print("num words %d" % len(self.id2voc))
        self._load_char_vocabulary()
        self._load_special_tokens()


        print("num words %d" %len(self.id2voc))

        self.voc_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(self.id2voc),
                                                                   default_value=self.unk_token_id)
        self.char_voc_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(self.char_id2voc),
                                                                        default_value=self.unk_char_id)

        #add chunk index as an additional feature
        self.load_chunk_index_tokens()
        # print(self.id2chunk_index_token)
        # print(self.char_id2voc)
        # exit(1)
        self.chunk_index_table = tf.contrib.lookup.index_table_from_tensor(tf.constant(self.id2chunk_index_token),
                                                                           default_value=self.unk_chunk_index_token_id)

    def _load_vocabulary(self):
        print("load vocabulary...")
        self.voc_dic = {}
        self.id2voc = []
        print(self.config.voc_path)
        with tf.gfile.GFile(self.config.voc_path, "r") as f:
            i = 0
            for line in f:
                word = line.split(" ")[0]
                self.voc_dic[word] = i
                self.id2voc.append(word)
                i += 1
        # print(self.voc_dic)
        # print(self.id2voc)


    def load_chunk_index_tokens(self):
        self.chunk_index_token = {}
        self.id2chunk_index_token = []
        for i in chunk_index_tokens:
            self.chunk_index_token[i] = len(self.id2chunk_index_token)
            self.id2chunk_index_token.append(i)

        self.num_chunk_index_tokens = len(self.id2chunk_index_token)
        self.unk_chunk_index_token = unknown_chunk_index_token
        self.unk_chunk_index_token_id = self.chunk_index_token[self.unk_chunk_index_token]


    def _load_special_tokens(self):
        # special_tokens = ["patterngo", "patterneos", "patternunk", "patternpad"]
        self.special_tokens = useful_tags

        for i in self.special_tokens:
            self.voc_dic[i] = len(self.id2voc)
            self.id2voc.append(i)

        self.num_special_tokens = len(self.special_tokens)

        # self.start_token_id = self._get_special_token_id("patterngo")
        # self.end_token_id = self._get_special_token_id("patterneos")
        self.unk_token_id = self._get_special_token_id("patternunk")
        self.pad_token_id = self._get_special_token_id("patternpad")

        self.pad_token = "patternpad"
        self.unk_char_id = self._get_special_char_id("charunk")
        self.pad_char_id = self._get_special_char_id("charpad")
        self.pad_char = "charpad"

    #=========process a single line
    def _process_list_of_indices(self, s):

        indices = tf.string_to_number(tf.string_split([s], delimiter=' ').values, out_type=tf.int32)
        length = tf.shape(indices)[0]

        return indices, length

    def _process_single_line(self, line):

        line_segs = tf.string_split([line], delimiter='\t')

        query = tf.string_split([line_segs.values[0]], delimiter=' ').values
        query_len = tf.shape(query)[0]

        # print("query_len", query_len.get_shape().as_list())
        # exit(1)
        chunk_index = line_segs.values[1]
        url = line_segs.values[2]
        header_index, h_length = self._process_list_of_indices(line_segs.values[3])
        fb_index, fb_length = self._process_list_of_indices(line_segs.values[4])
        sb_index, sb_length = self._process_list_of_indices(line_segs.values[5])
        tb_index, tb_length = self._process_list_of_indices(line_segs.values[6])
        # header_index = tf.string_to_number(line.values[1], out_type=tf.int32)
        #
        # print(header_index.get_shape())
        # exit(1)

        sents_ws = tf.string_split(line_segs.values[7:], delimiter=' ')
        sents_ws_shape = tf.cast(tf.shape(sents_ws), tf.int64)
        sents_used_for_chars = tf.sparse_to_dense(sparse_indices=sents_ws.indices,
                                                  output_shape=sents_ws_shape,
                                                  sparse_values=sents_ws.values,
                                                  default_value="")  # padded by empty string

        sents = tf.sparse_to_dense(sparse_indices=sents_ws.indices,
                                   output_shape=sents_ws_shape,
                                   sparse_values=sents_ws.values,
                                   default_value=self.pad_token)
        # sents removed by max length
        # print("sents shape ", sents.get_shape())
        # sents = tf.cond(tf.shape(sents)[0] > self.config.max_char_length,
        #                 lambda: sents[0:self.config.max_char_length, :],
        #                 lambda: sents)

        # truncate by max sent length
        sents_used_for_chars = sents_used_for_chars[:, 0:self.config.max_sent_length]
        sents = sents[:, 0:self.config.max_sent_length]

        # calculate sent length
        mask = tf.where(tf.equal(sents, self.pad_token),
                        tf.zeros(shape=tf.shape(sents), dtype=tf.int32),
                        tf.ones(shape=tf.shape(sents), dtype=tf.int32))
        sents_len = tf.reduce_sum(mask, axis=1)

        # deal with char level
        query2chars, sents2chars = self._token_to_chars(query, sents_used_for_chars, self.config.max_char_length,
                                                        self.pad_char)
        # print(header_index.get_shape())
        # exit(1)
        # map negative example

        # sents2words = tf.constant(sents2words, dtype=tf.string)
        # sents_len = tf.constant(sents_len, dtype=tf.int32)
        return {"raw_query": line_segs.values[0],
                "raw_sents": line_segs.values[7:],
                "raw_url": url,
                "query": query,
                "chunk_index": chunk_index,
                "header_index": header_index,
                "fb_index": fb_index,
                "sb_index": sb_index,
                "tb_index": tb_index,
                "header_length": h_length,
                "fb_length": fb_length,
                "sb_length": sb_length,
                "tb_length": tb_length,
                "sents2words": sents,
                "query_len": query_len,
                "sents_len": sents_len,
                "query2chars": query2chars,
                "sents2chars": sents2chars
                }
        # print(header_index)
        # return [sents]

    @property
    def _padded_shape(self):
        return {"raw_query": tf.TensorShape([]),
                "raw_sents": tf.TensorShape([None]),
                "raw_url": tf.TensorShape([]),
                "query": tf.TensorShape([None]),
                "chunk_index": tf.TensorShape([]),
                "header_index": tf.TensorShape([None]),
                "fb_index": tf.TensorShape([None]),
                "sb_index": tf.TensorShape([None]),
                "tb_index": tf.TensorShape([None]),
                "header_length": tf.TensorShape([]),
                "fb_length": tf.TensorShape([]),
                "sb_length": tf.TensorShape([]),
                "tb_length": tf.TensorShape([]),
                "sents2words": tf.TensorShape([None, None]),
                "query_len": tf.TensorShape([]),
                "sents_len": tf.TensorShape([None]),
                "query2chars": tf.TensorShape([None, self.config.max_char_length]),
                "sents2chars": tf.TensorShape([None, None, self.config.max_char_length])
                }
        # return [tf.TensorShape([None, None])]

    @property
    def _padded_values(self):
        return {"raw_query": "",
                "raw_sents": "",
                "raw_url": "",
                "query": self.pad_token,
                "chunk_index": self.unk_chunk_index_token,
                "header_index": 0,
                "fb_index": 0,
                "sb_index": 0,
                "tb_index": 0,
                "header_length": 0,
                "fb_length": 0,
                "sb_length": 0,
                "tb_length": 0,
                "sents2words": self.pad_token,
                "query_len": 0,
                "sents_len": 0,
                "query2chars": self.pad_char,
                "sents2chars": self.pad_char
                }

        # return [self.pad_token]

    def _get_example_len(self, input_dic):
        sents_len = input_dic["sents_len"]
        num_sents = tf.shape(sents_len)[0]
        return num_sents

    def _post_process(self, batch):
        output = {}
        # l = list(range(self.config.batch_size))
        # indices_tensor = tf.expand_dims(tf.constant(l, tf.int32), axis=1)

        # actual_batch_size = tf.shape(batch["raw_sents"])[0]
        # indices_tensor = tf.expand_dims(tf.range(actual_batch_size, delta=1, dtype=tf.int32), axis=1)
        # h_index = tf.concat([batch_index, tf.cast(tf.expand_dims(input_dic["adjust_p1"], axis=1), tf.int32)], axis=-1)
        # fb_index = tf.concat([batch_index, tf.cast(tf.expand_dims(input_dic["adjust_p2"], axis=1), tf.int32)], axis=-1)
        output["raw_query"] = batch["raw_query"]
        output["raw_sents"] = batch["raw_sents"]
        output["raw_url"] = batch["raw_url"]

        # some data for debug
        output["orig_query"] = batch["query"]
        output["orig_sents"] = batch["sents2words"]
        output["orig_query2chars"] = batch["query2chars"]
        output["orig_sents2chars"] = batch["sents2chars"]

        # output["raw_query"] = batch["query"]
        # output["raw_sents"] = batch["sents2words"]
        output["query"] = self.voc_table.lookup(batch["query"])
        output["sents2words"] = self.voc_table.lookup(batch["sents2words"])
        output["query_len"] = tf.reshape(tf.squeeze(batch["query_len"]), shape=[-1])  # give a static shape
        output["sents_len"] = batch["sents_len"]

        # don't allow a sents with length 0. Set it to 1
        # sents_len_shape= tf.shape(output["sents_len"])
        # output["sents_len"] = tf.where(tf.equal(output["sents_len"], tf.zeros(shape=sents_len_shape, dtype=tf.int32)),
        #                                tf.ones(shape=sents_len_shape, dtype=tf.int32),
        #                                output["sents_len"])

        # set all len to max
        # max_len = tf.reduce_max(output["sents_len"])
        # output["sents_len"] = tf.fill(dims=tf.shape(output["sents_len"]), value=max_len)
        output["query2chars"] = self.char_voc_table.lookup(batch["query2chars"])
        output["sents2chars"] = self.char_voc_table.lookup(batch["sents2chars"])

        #for chunk index
        output["chunk_index"] = self.chunk_index_table.lookup(batch["chunk_index"])

        tags = ["header", "fb", "sb", "tb"]
        for i in tags:
            output["%s_index"%i] = batch["%s_index"%i]
            output["%s_length" % i] = batch["%s_length" % i]

        return output

    def _filter_by_length(self, data):
        min_len = self.config.filter.min_len
        max_len = self.config.filter.max_len

        print("filtering data by (%d, %d)" % (min_len, max_len))

        def fn(input_dic):
            len = self._get_example_len(input_dic)
            return tf.logical_and(tf.greater(len, min_len), tf.less(len, max_len))

        return data.filter(fn)

    def _filter_by_query_length(self, data):
        max_query_len = self.config.filter_query.max_len
        print("filtering data by query length %d" % (max_query_len))

        def fn_q(input_dic):
            q_len = input_dic["query_len"]
            return tf.less(q_len, max_query_len)

        return data.filter(fn_q)

    def get_batch(self, num_batches=1):
        self.data = tf.data.TextLineDataset(self.config.data)

        self.data = self.data.map(self._process_single_line)

        if self.config.filter.enable:
            self.data = self._filter_by_length(self.data)

        if self.config.filter_query.enable:
            self.data = self._filter_by_query_length(self.data)

        # self.debug_dataset(self.data)
        if self.config.bucket.enable:
            print("bucketing is enabled")
            bucket_fn = tf.contrib.data.bucket_by_sequence_length(element_length_func=self._get_example_len,
                                                                  bucket_boundaries=self.config.bucket.lengths,
                                                                  bucket_batch_sizes=self.config.bucket.batch_sizes,
                                                                  padded_shapes=self._padded_shape,
                                                                  padding_values=self._padded_values,
                                                                  pad_to_bucket_boundary=False)
            self.data = self.data.apply(bucket_fn)

        else:
            self.data = self.data.padded_batch(self.config.batch_size,
                                               padded_shapes=self._padded_shape,
                                               padding_values=self._padded_values)

        self.data = self.data.map(self._post_process)
        # self.debug_dataset(self.data)
        # exit(1)


        if self.config.repeat:
            self.data = self.data.repeat(count=None)
            # self.data = self.data.shuffle(buffer_size=self.config.buffer_size)

        self.iter = self.data.make_initializable_iterator()

        # if num_batches == 1:
        #     batch = self.iter.get_next()
        #     return batch
        if num_batches > 0:
            batches = [self.iter.get_next() for _ in range(num_batches)]
            return batches
        else:
            raise Exception("number of batches is 0")

    def start(self, sess):

        sess.run(self.iter.initializer)

    def debug(self, ops):

        with tf.Session("list_loader") as sess:
            print(sess.run(ops))

    def debug_dataset(self, data):

        iter = data.make_initializable_iterator()
        example = iter.get_next()
        table_init_op = tf.tables_initializer()
        ops = [example["sents2words"], example["query_len"], example["tb_index"], example["tb_length"]]
        with tf.Session() as sess:
            sess.run(iter.initializer)
            sess.run(table_init_op)
            # print(sess.run(example))
            print(sess.run(example))

            # print(sess.run(example["sents_len"]))

        exit(1)

    # def _bounded_by_max_sent_length(self, sents, max_sent_length):
    #
    #     sents = sents[0:max_sent_length, :]
    #     return sents

    # char level
    def _load_char_vocabulary(self):

        print("load char vocabulary...")
        self.char_voc_dic = {}
        self.char_id2voc = []
        with tf.gfile.GFile(self.config.char_voc_path, "r") as f:
            i = 0
            for line in f:
                c = line.split(" ")[0]
                self.char_voc_dic[c] = i
                self.char_id2voc.append(c)
                i += 1

    def _token_to_chars(self, queries, sents, max_char_length, pad_char):

        print("chars representation")

        def process_one_token(token):
            chars = tf.string_split([token], delimiter="").values
            num_chars = tf.shape(chars)[0]
            return tf.cond(num_chars < max_char_length,
                           lambda: tf.pad(chars, paddings=[[0, max_char_length - num_chars]],
                                          constant_values=pad_char),
                           lambda: chars[0:max_char_length])

        # print(queries)
        # print(sents)

        query2chars = tf.map_fn(process_one_token, queries)
        query2chars = tf.reshape(query2chars, shape=[-1, max_char_length])

        sents_shape = tf.shape(sents)
        sents1d = tf.reshape(sents, shape=[-1])
        sents2chars = tf.map_fn(process_one_token, sents1d)
        num_sents = sents_shape[0]
        sents2chars = tf.reshape(sents2chars, shape=[num_sents, -1, max_char_length])

        return query2chars, sents2chars







