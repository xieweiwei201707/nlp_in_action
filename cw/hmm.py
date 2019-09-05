import os
import pickle


class HMM(object):
    def __init__(self):
        """
        初始化一些全局信息，用于初始化一些成员变量。如状态集合（标记S、B、E、M），以及存储概率计算的中间文件
        """

        # 主要是用于存储算法中间结果，不用每次都训练模型
        self.model_file = './data/hmm_model.pkl'

        # 状态值集合
        self.state_list = ['B', 'M', 'E', 'S']

        # 参数加载，用于判断是否需要重新加载model_file
        self.load_para = False

        # 状态转移概率（状态 -> 状态的条件概率）
        self.A_dic = dict()

        # 发射概率（状态 -> 词语的条件概率）
        self.B_dic = dict()

        # 状态的初始概率
        self.Pi_dic = dict()

    def try_load_module(self, trained):
        """
        接收一个参数，用于判断是否加载中间结果文件。当直接加载中间结果时，可以不通过语料库训练，直接进行分词调用。
        否则，该函数用于初始化初始概率、转移概率以及发射概率等信息。
        这里的初始概率是指，一句话第一个字被标记成"S""B""E"或"M"的概率
        用于加载已计算的中间结果，当需要重新训练时，需要初始化清空结果
        :param trained:
        :return:
        """
        if trained:
            with open(self.model_file, 'rb') as f:
                self.A_dic = pickle.load(f)
                self.B_dic = pickle.load(f)
                self.Pi_dic = pickle.load(f)
                self.load_para = True
        else:
            # 状态转移概率（状态 -> 状态的条件概率）
            self.A_dic = dict()

            # 发射概率（状态 -> 词语的条件概率）
            self.B_dic = dict()

            # 状态的初始概率
            self.Pi_dic = dict()
            self.load_para = False
        pass

    def train(self, path):
        """
        主要用于通过给定的分词语料进行训练。语料的格式为每行一句话（这里以逗号隔开也算一句），每个词以空格风格。
        该函数主要通过对语料的统计，得到HMM所需的初始概率、转移概率以及发射概率。
        计算转移概率、发射概率以及初始概率
        :param path:
        :return:
        """

        # 重置几个概率矩阵
        self.try_load_module(False)

        # 统计状态出现次数，求p(o)
        count_dic = {}

        # 初始化参数
        def init_parameters():
            for state in self.state_list:
                self.A_dic[state] = {s: 0.0 for s in self.state_list}
                self.Pi_dic[state] = 0.0
                self.B_dic[state] = {}
                count_dic[state] = 0

        def make_label(text):
            out_text = []
            if len(text) == 1:
                out_text.append('S')
            else:
                out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']

            return out_text

        init_parameters()
        line_num = -1

        # 观察者集合，主要是字以及标点等
        words = set()
        with open(path, encoding='utf8') as f:
            for line in f:
                line_num += 1

                line = line.strip()
                if not line:
                    continue

                word_list = [i for i in line if i != ' ']
                # 更新字的集合
                words |= set(word_list)

                line_list = line.split()

                line_state = []
                for w in line_list:
                    line_state.extend(make_label(w))

                if len(word_list) != len(line_state):
                    print(len(word_list), len(line_state))
                    assert len(word_list) == len(line_state)

                for k, v in enumerate(line_state):
                    count_dic[v] += 1
                    if k == 0:
                        # 每个句子的第一个字的状态，用于计算初始状态概率
                        self.Pi_dic[v] += 1
                    else:
                        # 计算转移概率
                        self.A_dic[line_state[k - 1]][v] += 1
                        # 计算发射概率
                        self.B_dic[line_state[k]][word_list[k]] = self.B_dic[line_state[k]].get(word_list[k], 0) + 1.0
        self.Pi_dic = {k: v * 1.0 / line_num for k, v in self.Pi_dic.items()}
        self.A_dic = {k: {k1: v1 / count_dic[k] for k1, v1 in v.items()}
                      for k, v in self.A_dic.items()}

        # 加1平滑
        self.B_dic = {k: {k1: (v1 + 1) / count_dic[k] for k1, v1 in v.items()}
                      for k, v in self.B_dic.items()}

        with open(self.model_file, 'wb') as f:
            pickle.dump(self.A_dic, f)
            pickle.dump(self.B_dic, f)
            pickle.dump(self.Pi_dic, f)

        return self

    def viterbi(self, text, states, start_p, trans_p, emit_p):
        """
        Viterbi算法的实现，是基于动态规划的一种实现，主要是求最大概率的路径。
        其输入参数为初始概率、转移概率以及发射概率，加上需要切分的句子
        :param text:
        :param states:
        :param start_p:
        :param trans_p:
        :param emit_p:
        :return:
        """
        v = [{}]
        path = {}
        for y in states:
            v[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]

        for t in range(1, len(text)):
            v.append({})
            newpath = {}

            # 检验训练的发射概率矩阵中是否有该字
            neverSeen = text[t] not in emit_p['S'].keys() and \
                text[t] not in emit_p['M'].keys() and \
                text[t] not in emit_p['E'].keys() and \
                text[t] not in emit_p['B'].keys()

            for y in states:
                # 设置未知字单独成词
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
                (prob, state) = max(
                    [(v[t-1][y0] * trans_p[y0].get(y, 0) * emitP, y0)
                     for y0 in states if v[t-1][y0] > 0])
                v[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(v[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(v[len(text) - 1][y], y) for y in states])

        return prob, path[state]

    def cut(self, text):
        """
        用于切词，其通过加载中间文件，调用veterbi函数来完成
        :param text:
        :return:
        """
        if not self.load_para:
            self.try_load_module(os.path.exists(self.model_file))

        text = text.strip()
        prob, pos_list = self.viterbi(text, self.state_list, self.Pi_dic, self.A_dic, self.B_dic)
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i+1]
                next = i+1
            elif pos == 'S':
                yield char
                next = i+1

        if next < len(text):
            yield text[next:]


if __name__ == '__main__':
    hmm = HMM()
    hmm.train('./data/trainCorpus.txt_utf8')

    text = u' 这是一个非常棒的方案！'
    res = hmm.cut(text)
    print(text)
    print(str(list(res)))

