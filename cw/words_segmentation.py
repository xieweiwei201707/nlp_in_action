class MM(object):
    """
    基于规则匹配进行分词实现
    """
    def __init__(self, dict_path):
        self.dictionary = set()
        self.maximum = 0
        # 读取词典
        with open(dict_path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.dictionary.add(line)
                if len(line) > self.maximum:
                    self.maximum = len(line)

    def cut(self, text):
        """
        按规则对文本进行切分。
        :param text: 待切分的文本
        :return: 切分后的词汇列表
        """
        raise NotImplementedError()


class IMM(MM):
    """
    逆向最大匹配实现
    """

    def cut(self, text):
        result = []
        index = len(text)
        while index > 0:
            word = None
            for size in range(self.maximum, 0, -1):
                if index - size < 0:
                    continue
                piece = text[(index - size):index]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index -= size
                    break

            if word is None:
                index -= 1

        return result[::-1]


class FMM(MM):
    """
    正向最大匹配实现
    """
    def cut(self, text):
        result = []
        length = len(text)
        index = 0
        while index < length:
            word = None
            for size in range(self.maximum, 0, -1):
                if index + size > length:
                    continue
                piece = text[index:(index + size)]
                if piece in self.dictionary:
                    word = piece
                    result.append(word)
                    index += size
                    break

            if word is None:
                index += 1

        return result[::-1]


class BiMM(MM):
    """
    双向最大匹配实现
    """

    def __init__(self, dict_path):
        super(BiMM, self).__init__(dict_path=dict_path)
        self.imm_tokenizer = IMM(dict_path)
        self.fmm_tokenizer = FMM(dict_path)

    def cut(self, text):
        imm_result = self.imm_tokenizer.cut(text)
        fmm_result = self.fmm_tokenizer.cut(text)

        if len(fmm_result) > len(imm_result):
            print('use FMM result')
            return fmm_result
        else:
            print('use IMM result')
            return imm_result


def main():
    text = "南京市长江大桥"

    tokenizer = IMM('./data/imm_dic.utf8')
    print(tokenizer.cut(text))

    tokenizer = FMM('./data/imm_dic.utf8')
    print(tokenizer.cut(text))

    tokenizer = BiMM('./data/imm_dic.utf8')
    print(tokenizer.cut(text))


if __name__ == '__main__':
    main()
