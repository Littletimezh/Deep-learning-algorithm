class rnn_python():

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4 ):#
        #初始化，给定输入句子x的长度word_dim，每个细胞层的隐藏神经元个数hidden_dim，默认为100，这也是s(t)和h(t)的维度。
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = 4
        #这一步的关键是理解UVW的维度，对于后面的计算有用
        self.U = np.random.uniform(-1 / np.sqrt(word_dim), 1 / np.sqrt(word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-1/np.sqrt(hidden_dim), 1/np.sqrt(hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-1 / np.sqrt(hidden_dim), 1 / np.sqrt(hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T+1, self.hidden_dim))#最后一行是0，为了对应第一个时刻的前一时刻，全部为0
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))
        for t in range(T):
            s[t] = np.tanh(self.W.dot(s[t-1]) + self.U.dot(x(t)))#这个s(t)是推导中的h(t)，注意转换思路
            o[t] = self.softmax(self.V.dot(s[t]))#softmax是自己定义的，比较简单我省略了。
        return [o, s]

    def predict(self, x):#根据输入预测下一个词的输出
        o = self.forward_propagation(x)[0]
        y_predict = np.argmax(o, axis=1)
        return y_predict

    def cross_entropy(self, x, y):#x0-1值与概率值相乘取对数，只关系1对应的概率就可以了
        L = 0
        N = np.sum(len(y_i) for y_i in y)
        for i in range(len(y)):#先暂时理解y为词的下标
            o = self.forward_propagation(x[i])[0]
            correct_loss = o[:, y[i]]#每一行代表每一词的求和，但是要注意这里的y[i]是一个index区别于公式中y，公式中的y还是1
            L -= np.sum(np.log(correct_loss))
        return L/N

    def bptt(self, x, y):
        T = len(y)
        V = self.V
        W = self.W
        U = self.U
        o, s = self.forward_propagation(x)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        dLdU = np.zeros(self.U.shape)
        deltao = o
        deltao[:, y] -= 1#推导过程的y-o
        for t in range(T, -1, -1):#反向传播从第T个开始
            dLdV += np.outer(deltao[t], s(t).T)#一句话所有文字
            delta_t = self.V.T.dot(deltao[t]) * (1-s[t] ** 2)
            for bptt_step in np.range(t+1, max(0, t - self.bptt_truncate), -1):
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU += np.outer(delta_t, x[t])
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdW, dLdU, dLdV]

    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("./data", one_hot=True)

    ##参数设置，这些参数在建立模型时候需要赋予，一般都是默认的，当然也可以进行参数优化
    learning_rate = 0.01  # 学习率
    train_step = 10000  # 训练步数
    batch_size = 128  # 每次训练样本个数
    display_step = 10  # 每几次打印一次结果
    frame_size = 28  # 可以理解为一个时间序列的长度，图片一般为28*28，理解为每一行相互关联，时间t的长度T
    sequence_length = 28  # 序列长度，注意输入长度
    hidden_num = 5  # 细胞单元的神经元大小，也是每层的输出大小
    n_classes = 10  # 分类个数，一般为2分类问题,此处案例为二分类任务

    ##定义输入、输出，注意这部分是固定的我们事先知道的，无需训练的，但是需要我们传给模型，
    # 但在建立模型的时候我们并不知道传入什么、传入多少，因此建一个占位符，等到要用的时候好传入
    # 常数
    x = tf.placeholder(dtype=tf.float64, shape=[None, frame_size * sequence_length],
                       name='input_x')  # None代表提前也不知道输入几个样本训练
    y = tf.placeholder(dtype=tf.float64, shape=[None, n_classes], name='output_y')  # 这个y代表最终的输出，和rnn细胞的输出区分开
    # 变量
    weights = tf.Variable(tf.truncated_normal(shape=[hidden_num, n_classes]))  # 截尾正态分布初始化
    bias = tf.Variable(tf.zeros(shape=hidden_num))

    # 定义函数

    def RNN(x, weights, bias):
        x = tf.reshape(x, shape=[-1, frame_size, sequence_length])  # tf所有进模型的张量都是三维的,-1代表根据输入确定,缺省值根据计算获得
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)  # 只需传hidden_num
        init_state = tf.zeros(shape=[batch_size, rnn_cell.state_size])
        # 注state是rnn和lstm的专用实际表示细胞层的输出,注意init_state的问题
        output, states = tf.nn.dynamic_rnn(rnn_cell, x,
                                           dtype=tf.float32)  # call函数每次计算一步，tf.nn.dynamic_mn函数相当于调用n次call函数
        # output是每一层的output, states是最后一层的稳定输出。
        return tf.nn.softmax(tf.matmul(output[:, -1, :], weights) + bias, 1)  # 取0-1

    predy = RNN(x, weights, bias)
    lost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy, y))
    train = tf.train.AdamOptimizer(learning_rate).minimize(lost)
    correct_predict = tf.equal(tf.argmax(predy, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.to_float(correct_predict))

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    step = 1
    test_x, test_y = mnist.text.next_batch(batch_size)
    while step < train_step:
        batch_x, batch_y = mnist.text.next_batch(batch_size)
        batch_x = tf.reshape(batch_x, shape=[batch_size, frame_size, sequence_length])  # 训练样本，时长，句子长度
        if step % train_step == 0:
            acc, loss = sess.run([accuracy, lost], feed_dict=(batch_x, batch_y))
            print(step, acc, loss)
        step += 1