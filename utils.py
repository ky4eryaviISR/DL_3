import enum
import matplotlib.pyplot as plt


class PRINT(enum.Enum):
    TRAIN_ACC = 1
    TRAIN_LSS = 2
    TEST_LSS = 3
    TEST_ACC = 4


to_print = {PRINT.TEST_ACC: [],
            PRINT.TEST_LSS: [],
            PRINT.TRAIN_ACC: [],
            PRINT.TRAIN_LSS: []}


def save_graph(train,test,y_axis):
    plt.suptitle(y_axis, fontsize=20)
    plt.figure()
    plt.plot(train,color='r', label='train')
    plt.plot(test, color='g', label='test')
    plt.xlabel('Epochs')
    plt.legend(loc="upper left")
    plt.ylabel(y_axis)
    plt.savefig(y_axis+'.png')
